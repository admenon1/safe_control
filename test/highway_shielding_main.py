import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Add gatekeeper directory to path
gatekeeper_path = os.path.join(os.path.dirname(__file__), '../../gatekeeper')
sys.path.insert(0, gatekeeper_path)

from shielding import Gatekeeper  # Using shielding implementation
from robots.robot import BaseRobot

"""
Created on November 17th, 2025
@author: Integration of Shielding with Highway Environment
============================================
Highway scenario using Shielding algorithm for safe control.
Shielding ensures backup controller is always available from current state.
"""


# ========================================================================
# Highway Environment (Visualization)
# ========================================================================
class HighwayEnv:
    def __init__(self, width=60.0, height=12.0, num_lanes=3):
        self.width = width
        self.height = height
        self.num_lanes = num_lanes
        self.lane_width = height / num_lanes
        self.lane_centers = [
            (i + 0.5) * self.lane_width 
            for i in range(num_lanes)
        ]

    def render_lanes(self, ax, ego_x=None):
        """Draw lane markings and update view to follow ego vehicle."""
        created = []

        # Follow ego vehicle
        if ego_x is not None:
            margin = 15.0
            ax.set_xlim(ego_x - margin, ego_x + margin)

        x0, x1 = ax.get_xlim()

        # Draw lane boundaries
        for i in range(self.num_lanes + 1):
            y = i * self.lane_width
            line1, = ax.plot([x0, x1], [y, y], color='black', linewidth=1.0, zorder=0)
            line2, = ax.plot([x0, x1], [y, y], color='blue', linestyle='--', linewidth=0.6, zorder=0)
            created.append(line1)
            created.append(line2)

        return created


# ========================================================================
# Highway Controller with Shielding
# ========================================================================
class HighwayShieldingController:
    def __init__(self, X0, robot_spec, highway_env, dt=0.05, 
                 nominal_horizon=2.0, backup_horizon=4.0, event_offset=1.0,
                 show_animation=True, save_animation=False, ax=None, fig=None):
        self.highway_env = highway_env
        self.dt = dt
        self.show_animation = show_animation
        self.save_animation = save_animation
        self.ax = ax or plt.axes()
        self.fig = fig or plt.figure()

        # Robot setup
        self.robot_spec = robot_spec
        self.robot = BaseRobot(X0.reshape(-1, 1), robot_spec, dt, self.ax)

        # Traffic and obstacles
        self._traffic = []
        self._traffic_patches = []
        self._lane_patches = []
        self._nominal_traj_lines = []
        self._backup_traj_lines = []
        self.obs = np.empty((0, 7))

        # Shielding controller (using Gatekeeper class from shielding.py)
        self.shielding = Gatekeeper(
            self.robot, 
            dt=dt, 
            nominal_horizon=nominal_horizon, 
            backup_horizon=backup_horizon, 
            event_offset=event_offset
        )
        
        # Enable backup trajectory visualization
        self.shielding.visualize_backup = robot_spec.get('visualize_backup_set', True)
        
        # Set nominal and backup controllers
        self.target_lane_idx = robot_spec.get('target_lane_index', 2)
        self.target_lane = highway_env.lane_centers[self.target_lane_idx]
        
        # Override robot's nominal controller with highway-specific version
        self.robot.nominal_input = self.nominal_input_highway
        
        # Set controllers in Shielding
        self.shielding._set_nominal_controller(self.nominal_controller_wrapper)
        self.shielding._set_backup_controller(self.backup_controller_wrapper)

        # Waypoints and control reference
        self.waypoints = None
        self.goal_reached_flag = False
        self.control_ref = None  # Will be set when waypoints are set

    def nominal_controller_wrapper(self, state, goal):
        """
        Wrapper for Shielding to call nominal controller.
        Shielding expects: nominal_controller(state, goal) -> control (nu, 1)
        """
        return self.nominal_input_highway(target_speed=self.robot_spec.get('v_nominal', 2.0))

    def backup_controller_wrapper(self, state):
        """
        Wrapper for Shielding to call backup controller (lane change).
        Shielding expects: backup_controller(state) -> control (nu, 1)
        """
        return self.backup_lane_change_control(state)

    def nominal_input_highway(self, target_speed=2.0, **kwargs):
        """
        Nominal controller for highway driving: constant speed cruise with heading alignment.
        
        Args:
            target_speed: desired forward velocity
        
        Returns:
            np.array([[accel], [omega]]): control input
        """
        X = self.robot.X
        v = X[3, 0]
        theta = X[2, 0]
        
        # Controller gains
        k_a = 1.0          # Proportional gain for acceleration
        k_omega = 0.5      # Proportional gain for yaw rate
        target_heading = 0.0  # Drive straight (horizontal)
        
        # Speed regulation (P control on velocity error)
        accel = k_a * (target_speed - v)
        
        # Heading alignment (P control on angle error)
        heading_error = target_heading - theta
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        omega = k_omega * heading_error
        
        # Clip to limits
        accel = np.clip(accel, -self.robot_spec['a_max'], self.robot_spec['a_max'])
        omega = np.clip(omega, -self.robot_spec['w_max'], self.robot_spec['w_max'])
        
        return np.array([[accel], [omega]])

    def backup_lane_change_control(self, state):
        """
        Backup controller: PD-based lane change to target lane.
        
        Args:
            state: current state [px, py, theta, v] (nx, 1) or (nx,)
        
        Returns:
            u_b: backup control [[accel], [omega]]
        """
        state = state.reshape(-1)
        px, py, theta, v = state[:4]

        # Backup parameters
        backup_time = 4.0
        zeta = 0.7
        omega_n = 4.0 / (zeta * backup_time)
        
        # PD gains
        k_y = omega_n**2
        k_psi = 2.0 * zeta * omega_n
        
        v_safe = max(abs(v), 0.5)
        omega_max = self.robot_spec.get('w_max', 0.5)
        
        # Errors (straight road, lane aligned with x-axis)
        e_y = py - self.target_lane
        e_psi = theta
        
        # PD yaw-rate control
        omega = -k_psi * e_psi - (k_y / v_safe) * e_y
        omega = np.clip(omega, -omega_max, omega_max)
        
        # No longitudinal acceleration during lane change
        accel = 0.0
        return np.array([[accel], [omega]])

    def init_traffic(self, traffic_spec):
        """Initialize traffic cars with positions and velocities."""
        if not traffic_spec:
            self._traffic = []
            self._traffic_patches = []
            self.obs = np.empty((0, 7))
            return

        rows = []
        patches_list = []
        for car in traffic_spec:
            rows.append([
                car["x"], car["y"], car["radius"],
                car.get("vx", 0.0), car.get("vy", 0.0), 0.0, 0.0
            ])
            patch = self.ax.add_patch(
                patches.Circle(
                    (car["x"], car["y"]),
                    car["radius"],
                    facecolor="blue",
                    edgecolor="black",
                    alpha=0.7,
                    zorder=3
                )
            )
            patches_list.append(patch)

        self._traffic = traffic_spec
        self._traffic_patches = patches_list
        self.obs = np.array(rows, dtype=float)

    def _advance_traffic(self):
        """Update traffic car positions based on their velocities."""
        if not self._traffic:
            return

        for idx, car in enumerate(self._traffic):
            car["x"] += car.get("vx", 0.0) * self.dt
            car["y"] += car.get("vy", 0.0) * self.dt
            self.obs[idx, 0] = car["x"]
            self.obs[idx, 1] = car["y"]

            if idx < len(self._traffic_patches):
                self._traffic_patches[idx].center = (car["x"], car["y"])

    def set_waypoints(self, waypoints):
        """Set goal waypoints for the ego vehicle and initialize control reference."""
        self.waypoints = np.array(waypoints, dtype=float)
        
        # Initialize control reference (defined once, not in loop)
        goal = self.waypoints[-1] if self.waypoints is not None else np.array([55.0, self.highway_env.lane_centers[0]])
        self.control_ref = {
            'goal': goal.reshape(-1, 1),
            'state_machine': 'track',
            'u_ref': np.zeros((2, 1))
        }

    def has_reached_goal(self):
        """Check if ego vehicle reached the final waypoint."""
        if self.waypoints is None or len(self.waypoints) == 0:
            return True
        ego_pos = self.robot.get_position()
        goal_pos = self.waypoints[-1, :2]
        return np.linalg.norm(ego_pos - goal_pos) < 2.0

    def control_step(self):
        """Execute one control loop iteration using Shielding."""
        self._advance_traffic()

        # Get safe control from Shielding (control_ref defined once in set_waypoints)
        u_safe = self.shielding.solve_control_problem(
            self.robot.X, 
            self.control_ref, 
            self.obs
        )

        # Step robot dynamics
        self.robot.step(u_safe.reshape(-1, 1))

        if self.show_animation:
            self.robot.render_plot()

        return 0

    def draw_plot(self, pause=0.01):
        """Update visualization (lanes, trajectories, etc.)."""
        if not self.show_animation:
            return

        # Clear previous lane patches
        for patch in self._lane_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._lane_patches.clear()

        # Clear previous trajectory lines
        for line in self._nominal_traj_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._nominal_traj_lines.clear()
        
        for line in self._backup_traj_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._backup_traj_lines.clear()

        # Draw backup trajectories from Shielding (starts from current state)
        if hasattr(self.shielding, 'visualize_backup') and self.shielding.visualize_backup:
            trajs = self.shielding.get_backup_trajectories()
            for backup_traj in trajs:
                line, = self.ax.plot(
                    backup_traj[:, 0], backup_traj[:, 1],
                    color='red', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2
                )
                self._backup_traj_lines.append(line)

        # Draw committed trajectory from Shielding
        if hasattr(self.shielding, 'committed_x_traj') and self.shielding.committed_x_traj is not None:
            traj = self.shielding.committed_x_traj
            
            # Separate nominal and backup portions
            if hasattr(self.shielding, 'committed_horizon'):
                nominal_len = int(self.shielding.committed_horizon / self.dt)
                
                # Draw nominal trajectory (green)
                if nominal_len > 0 and nominal_len <= len(traj):
                    line, = self.ax.plot(
                        traj[:nominal_len, 0], traj[:nominal_len, 1],
                        color='green', linestyle='-', linewidth=2.0, alpha=0.8, zorder=2,
                        label='Nominal trajectory'
                    )
                    self._nominal_traj_lines.append(line)
                
                # Draw backup trajectory (red for shielding)
                if nominal_len < len(traj):
                    line, = self.ax.plot(
                        traj[nominal_len:, 0], traj[nominal_len:, 1],
                        color='red', linestyle='--', linewidth=2.0, alpha=0.8, zorder=2,
                        label='Backup trajectory'
                    )
                    self._backup_traj_lines.append(line)

        # Render lanes
        ego_x = float(self.robot.get_position()[0])
        lane_patches = self.highway_env.render_lanes(self.ax, ego_x=ego_x)
        if lane_patches:
            self._lane_patches.extend(lane_patches)

        # Refresh plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(pause)


# ========================================================================
# Main Simulation Entry Point
# ========================================================================
def highway_shielding_main(save_animation=False):
    """Run the highway scenario with Shielding control."""
    # Create highway environment (3 lanes)
    env = HighwayEnv(width=60.0, height=12.0, num_lanes=3)

    # Setup plotting
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Highway Scenario with Shielding Algorithm")

    # Robot specification
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'radius': 1.0,
        'v_nominal': 2.0,
        'target_lane_index': 2,  # Target rightmost lane for backup
        'robot_id': 0,
        'visualize_backup_set': True,  # Enable backup trajectory visualization
    }

    # Initial state [x, y, θ, v] - start in lane 1 at nominal speed
    x0 = np.array([-2.0, env.lane_centers[0], 0.0, robot_spec['v_nominal']]).reshape(-1, 1)

    # Create controller with Shielding
    controller = HighwayShieldingController(
        X0=x0,
        robot_spec=robot_spec,
        highway_env=env,
        dt=0.05,
        nominal_horizon=2.0,
        backup_horizon=4.0,  # Backup horizon for lane change
        event_offset=0.5,     # Replan frequently for reactive shielding
        show_animation=True,
        save_animation=save_animation,
        ax=ax,
        fig=fig
    )

    # Set waypoint: drive straight in lane 1
    waypoints = np.array([[55.0, env.lane_centers[0], 0.0]])
    controller.set_waypoints(waypoints)

    # Add traffic obstacle in lane 2
    controller.init_traffic([
        {
            "x": 15.0,
            "y": env.lane_centers[1],
            "radius": 1.0,
            "vx": 0.5  # slower-moving obstacle
        }
    ])

    # Run simulation loop
    print("Starting highway scenario simulation with Shielding...")
    print(f"Nominal horizon: {controller.shielding.nominal_horizon}s")
    print(f"Backup horizon: {controller.shielding.backup_horizon}s")
    print(f"Event offset: {controller.shielding.event_offset}s")
    print("\nShielding: Backup always available from current state")
    
    try:
        step_count = 0
        while not controller.has_reached_goal():
            controller.control_step()
            controller.draw_plot()
            step_count += 1

            # Stop if ego reaches x = 55
            if controller.robot.X[0, 0] >= 55.0:
                break

            # Safety limit on simulation steps
            if step_count > 5000:
                print("Maximum simulation steps reached.")
                break

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("Simulation complete.")
        print(f"Total steps: {step_count}")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    highway_shielding_main(save_animation=False)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robots.robot import BaseRobot
from backup_cbf.backup_cbf_qp import BackupCBFQP

"""
Created on November 15th, 2025
@author: Aswin D Menon
============================================
Consolidated implementation of highway scenario using Backup CBF with double lane-change as the backup controller.
Integrates environment rendering, traffic simulation, and backup.

"""


# ========================================================================
# Highway Environment
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
# Highway Controller
# ========================================================================
class HighwayController:
    def __init__(self, X0, robot_spec, highway_env, dt=0.05, 
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
        self._backup_traj_lines = []
        self.obs = np.empty((0, 7))

        # Backup CBF-QP filter
        self.backup_cbf_filter = BackupCBFQP(self.robot, self.robot_spec)  #### Shielding comes here

        # Waypoints
        self.waypoints = None
        self.goal_reached_flag = False

        # Overriding functions in robot.py
        self.robot.nominal_input = self.nominal_input_constant_speed  ## this will remain unchanged for mps

    def nominal_input_constant_speed(self, target_speed=2.0, **kwargs):
        """
        Custom nominal controller for highway driving - overrides robot.nominal_input.
        Implements constant speed cruise with heading alignment.
        
        Args:
            target_speed: desired forward velocity
            **kwargs: catches any additional arguments (like 'goal' from base class)
        
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
        """Set goal waypoints for the ego vehicle."""
        self.waypoints = np.array(waypoints, dtype=float)

    def has_reached_goal(self):
        """Check if ego vehicle reached the final waypoint."""
        if self.waypoints is None or len(self.waypoints) == 0:
            return True
        ego_pos = self.robot.get_position()
        goal_pos = self.waypoints[-1, :2]
        return np.linalg.norm(ego_pos - goal_pos) < 2.0

    def control_step(self):
        """Execute one control loop iteration."""
        self._advance_traffic()

        # Nominal control: constant speed, heading alignment
        u_des = self.robot.nominal_input(
                target_speed=self.robot_spec.get('v_nominal', 2.0)
            ).flatten()

        # Find nearest obstacle
        if self.obs.size > 0:
            ego_pos = self.robot.get_position()
            dists = np.linalg.norm(self.obs[:, :2] - ego_pos, axis=1)
            guard = self.obs[np.argmin(dists)]
        else:
            guard = np.array([1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Apply backup Backup CBF filter
        u_safe, intervening = self.backup_cbf_filter.backup_cbf_qp(              #### this is where shielding is called
            self.robot.X.flatten(), u_des, guard
        )

        # Step robot dynamics
        self.robot.step(u_safe.reshape(-1, 1))

        if self.show_animation:
            self.robot.render_plot()

        return 0 if not intervening else 1

    def draw_plot(self, pause=0.01):
        """Update visualization (lanes, backup trajectories, etc.)."""
        if not self.show_animation:
            return

        # Clear previous lane patches
        for patch in self._lane_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._lane_patches.clear()

        # Clear previous backup trajectory lines
        for line in self._backup_traj_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._backup_traj_lines.clear()

        # Draw stored backup trajectories
        if hasattr(self.backup_cbf_filter, 'visualize_backup') and self.backup_cbf_filter.visualize_backup:
            trajs = self.backup_cbf_filter.get_backup_trajectories()
            for phi in trajs:
                line, = self.ax.plot(
                    phi[:, 0], phi[:, 1],
                    color='orange', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2
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
def highway_scenario_main(save_animation=False):
    """Run the highway scenario with backup CBF control."""
    # Create highway environment (3 lanes)
    env = HighwayEnv(width=60.0, height=12.0, num_lanes=3)

    # Setup plotting
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # Robot specification
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'backup_trigger_dist': 6.0,
        'backup_type': 'lane_change',
        'lane_centers': env.lane_centers,
        'backup_lane_index': 2,  # Target rightmost lane (lane 3)
        'radius': 1.0,
        'v_nominal': 2.0,
        'visualize_backup_set': True,
    }

    # Initial state [x, y, θ, v] - start in lane 1 at nominal speed
    x0 = np.array([-2.0, env.lane_centers[0], 0.0, robot_spec['v_nominal']]).reshape(-1, 1)

    # Create controller
    controller = HighwayController(
        X0=x0,
        robot_spec=robot_spec,
        highway_env=env,
        dt=0.05,
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
    print("Starting highway scenario simulation...")
    try:
        while not controller.has_reached_goal():
            controller.control_step()
            controller.draw_plot()

            # Stop if ego reaches x = 55
            if controller.robot.X[0, 0] >= 55.0:
                break

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("Simulation complete.")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    highway_scenario_main(save_animation=False)
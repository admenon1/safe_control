import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
from robots.robot import BaseRobot
from backup_cbf.backup_cbf_qp import BackupCBFQP
from gatekeeper.gatekeeper import Gatekeeper
from gatekeeper.shielding import Shielding

"""
Created on November 15th, 2025
@author: Aswin D Menon
Updated on November 18th, 2025
============================================
Modular highway scenario supporting multiple safety controllers:
- backup_cbf 
- gatekeeper
- shielding

Usage:
    python highway_main.py [controller_type]
    controller_type: 'backup_cbf' (default), 'gatekeeper', or 'shielding'
"""


# ========================================================================
# Highway Environment Parameters and Functions
# ========================================================================
class HighwayEnvironmentConfig:
    """
    Encapsulates all highway-scenario-specific parameters and CBF definitions.
    """
    def __init__(self, lane_centers, target_lane_idx=2):
        # Lane parameters
        self.lane_centers = lane_centers
        self.target_lane_idx = target_lane_idx
        self.target_lane = lane_centers[target_lane_idx]
        self.target_speed = 0.0
        self.lane_margin = 0.5

        # Backup controller parameters
        self.backup_time = 4.0  # Match backup horizon for responsive lane changes
        self.backup_zeta = 0.7
        self.omega_max_backup = 0.5

        # CBF class-K function parameters
        self.alpha_linear_coeff = 15.0
        self.alpha_b_coeff = 10.0


    # ------------------------------------------------------------------
    # Backup Controller (Lane Change)
    # ------------------------------------------------------------------
    def backup_control(self, x, robot_spec):
        """
        PD controller for lane change to target lane.
        Achieves ~T-second settling with ζ≈0.7.
        
        Args:
            x: state [px, py, theta, v]
            robot_spec: robot specification dict
        
        Returns:
            u_b: backup control [[accel], [omega]]
        """
        x = x.reshape(-1)
        px, py, theta, v = x[:4]

        # Desired settling time and damping
        T = max(self.backup_time, 1e-3)
        zeta = self.backup_zeta

        # Natural frequency from settling time Ts ≈ 4/(ζ ω_n)
        omega_n = 4.0 / (zeta * T)

        # PD gains
        k_y = omega_n**2
        k_psi = 2.0 * zeta * omega_n

        v_safe = max(abs(v), 0.5)
        omega_max = self.omega_max_backup

        # Errors (straight road, lane aligned with x-axis)
        e_y = py - self.target_lane
        e_psi = theta

        # PD yaw-rate control
        omega = -k_psi * e_psi - (k_y / v_safe) * e_y
        omega = np.clip(omega, -omega_max, omega_max)

        # No longitudinal acceleration during lane change
        accel = 0.0
        return np.array([[accel], [omega]])

    # ------------------------------------------------------------------
    # Safety CBF (Collision Avoidance)
    # ------------------------------------------------------------------
    def h_safety(self, x, obs, robot_radius):
        """
        Safety CBF: Distance to obstacle minus combined radius.
        h > 0 means safe, h = 0 is boundary, h < 0 is collision.
        
        Args:
            x: state [px, py, theta, v]
            obs: obstacle [ox, oy, radius, vx, vy, ...]
            robot_radius: ego robot radius
        
        Returns:
            h: CBF value (positive = safe)
        """
        px, py = x[:2]
        ox, oy, r = obs[:3]
        combined = r + robot_radius
        return (px - ox) ** 2 + (py - oy) ** 2 - combined ** 2

    def grad_h_safety(self, x, obs):
        """
        Gradient of safety CBF with respect to state.
        
        Returns:
            grad_h: gradient vector (nx,)
        """
        px, py = x[:2]
        ox, oy = obs[:2]
        grad = np.zeros(x.size)
        grad[0] = 2.0 * (px - ox)
        grad[1] = 2.0 * (py - oy)
        return grad

    # ------------------------------------------------------------------
    # Terminal CBF (Lane Reachability)
    # ------------------------------------------------------------------
    def h_backup(self, x):
        """
        Terminal CBF: Vehicle is within lane margin of target lane.
        
        Returns:
            h_b: CBF value (positive = in target lane)
        """
        _, py, _, v = x[:4]
        lane_err = py - self.target_lane
        return -0.5 * (lane_err ** 2 - self.lane_margin ** 2)

    def grad_h_backup(self, x):
        """
        Gradient of terminal CBF with respect to state.
        
        Returns:
            grad_h_b: gradient vector (nx,)
        """
        grad = np.zeros(x.size)
        grad[1] = -(x[1] - self.target_lane)
        grad[3] = -(x[3] - self.target_speed)
        return grad

    # ------------------------------------------------------------------
    # Class-K Functions (CBF Enforcement)
    # ------------------------------------------------------------------
    def alpha(self, h):
        """
        Class-K function for safety CBF.
        Controls aggressiveness of collision avoidance.
        
        Form: α(h) = c₁·h
        """
        return self.alpha_linear_coeff * h

    def alpha_b(self, h_b):
        """
        Class-K function for terminal CBF.
        Controls aggressiveness of lane-reaching.
        
        Form: α_b(h_b) = c·h_b
        """
        return self.alpha_b_coeff * h_b


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
# Highway Controller
# ========================================================================
class HighwayController:
    def __init__(self, X0, robot_spec, highway_env, highway_config, dt=0.05, 
                 controller_type='backup_cbf', nominal_horizon=2.0, backup_horizon=4.0, event_offset=0.5,
                 show_animation=True, save_animation=False, ax=None, fig=None):
        self.highway_env = highway_env
        self.highway_config = highway_config
        self.dt = dt
        self.controller_type = controller_type
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
        self._nominal_traj_lines = []
        self.obs = np.empty((0, 7))

        # Safety controller (selected based on controller_type)
        self.safety_controller = None
        self.control_ref = None  # For gatekeeper/shielding
        
        if controller_type == 'backup_cbf':
            self.safety_controller = BackupCBFQP(self.robot, self.robot_spec)
            self._configure_backup_cbf()
        elif controller_type in ['gatekeeper', 'shielding']:
            if controller_type == 'shielding':
                self.safety_controller = Shielding(
                    self.robot, dt=dt, 
                    nominal_horizon=nominal_horizon,
                    backup_horizon=backup_horizon,
                    event_offset=event_offset
                )
            else:  # gatekeeper
                print("Initializing Gatekeeper controller.")
                self.safety_controller = Gatekeeper(
                    self.robot, dt=dt,
                    nominal_horizon=nominal_horizon,
                    backup_horizon=backup_horizon,
                    event_offset=event_offset
                )
            
            # Enable visualization
            self.safety_controller.visualize_backup = robot_spec.get('visualize_backup_set', True)
            
            # Set controllers
            self.safety_controller._set_nominal_controller(self.nominal_controller_wrapper)
            self.safety_controller._set_backup_controller(self.backup_controller_wrapper)
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}. Choose 'backup_cbf', 'gatekeeper', or 'shielding'.")

        # Waypoints
        self.waypoints = None
        self.goal_reached_flag = False

        # Override robot's nominal controller
        self.robot.nominal_input = self.nominal_input_constant_speed

    def _configure_backup_cbf(self):
        """
        Configure the BackupCBFQP with highway-specific callbacks.
        This is where environment-specific logic is given.
        """
        # Create lambda wrappers to bind robot_spec
        backup_control_fn = lambda x: self.highway_config.backup_control(x, self.robot_spec)
        h_safety_fn = self.highway_config.h_safety
        grad_h_safety_fn = self.highway_config.grad_h_safety
        h_backup_fn = self.highway_config.h_backup
        grad_h_backup_fn = self.highway_config.grad_h_backup
        alpha_fn = self.highway_config.alpha
        alpha_b_fn = self.highway_config.alpha_b

        # Inject into BackupCBFQP
        self.safety_controller.set_environment_callbacks(
            backup_time=self.highway_config.backup_time,
            backup_control_fn=backup_control_fn,
            h_safety_fn=h_safety_fn,
            grad_h_safety_fn=grad_h_safety_fn,
            h_backup_fn=h_backup_fn,
            grad_h_backup_fn=grad_h_backup_fn,
            alpha_fn=alpha_fn,
            alpha_b_fn=alpha_b_fn
        )

    def nominal_controller_wrapper(self, state, goal):
        """Wrapper for Gatekeeper/Shielding to call nominal controller."""
        return self.nominal_input_constant_speed(target_speed=self.robot_spec.get('v_nominal', 2.0))

    def backup_controller_wrapper(self, state):
        """Wrapper for Gatekeeper/Shielding to call backup controller."""
        return self.highway_config.backup_control(state, self.robot_spec)

    def nominal_input_constant_speed(self, target_speed=2.0, **kwargs):
        """
        Nominal controller: Constant speed cruise with heading alignment.
        """
        X = self.robot.X
        v = X[3, 0]
        theta = X[2, 0]
        
        # Controller gains
        k_a = 1.0
        k_omega = 0.5
        target_heading = 0.0  # Drive straight

        # Speed regulation
        accel = k_a * (target_speed - v)
        
        # Heading alignment
        heading_error = target_heading - theta
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
        
        # Initialize control reference for gatekeeper/shielding
        if self.controller_type in ['gatekeeper', 'shielding']:
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
        """Execute one control loop iteration."""
        self._advance_traffic()

        if self.controller_type == 'backup_cbf':
            # Backup CBF-QP mode
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

            # Apply backup CBF filter
            u_safe, intervening = self.safety_controller.backup_cbf_qp(
                self.robot.X.flatten(), u_des, guard
            )
            
            # Step robot dynamics
            self.robot.step(u_safe.reshape(-1, 1))
            return_val = 0 if not intervening else 1
            
        elif self.controller_type in ['gatekeeper', 'shielding']:
            # Gatekeeper/Shielding mode
            u_safe = self.safety_controller.solve_control_problem(
                self.robot.X,
                self.control_ref,
                self.obs
            )
            
            # Step robot dynamics
            self.robot.step(u_safe.reshape(-1, 1))
            return_val = 0

        if self.show_animation:
            self.robot.render_plot()

        return return_val

    def draw_plot(self, pause=0.01):
        """Update visualization."""
        if not self.show_animation:
            return

        # Clear previous patches
        for patch in self._lane_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._lane_patches.clear()

        for line in self._backup_traj_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._backup_traj_lines.clear()
        
        for line in self._nominal_traj_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._nominal_traj_lines.clear()

        # Draw backup trajectories (common for all controllers)
        if hasattr(self.safety_controller, 'visualize_backup') and self.safety_controller.visualize_backup:
            trajs = self.safety_controller.get_backup_trajectories()
            
            # Different colors for different controllers
            color = 'orange' if self.controller_type in ['backup_cbf', 'gatekeeper'] else 'red'
            
            for traj in trajs:
                line, = self.ax.plot(
                    traj[:, 0], traj[:, 1],
                    color=color, linestyle='--', linewidth=1.0, alpha=0.7, zorder=2
                )
                self._backup_traj_lines.append(line)

        # Draw committed trajectory for gatekeeper/shielding
        if self.controller_type in ['gatekeeper', 'shielding']:
            if hasattr(self.safety_controller, 'committed_x_traj') and self.safety_controller.committed_x_traj is not None:
                traj = self.safety_controller.committed_x_traj
                
                # Separate nominal and backup portions
                if hasattr(self.safety_controller, 'committed_horizon'):
                    nominal_len = int(self.safety_controller.committed_horizon / self.dt)
                    
                    # Draw nominal trajectory (green)
                    if nominal_len > 0 and nominal_len <= len(traj):
                        line, = self.ax.plot(
                            traj[:nominal_len, 0], traj[:nominal_len, 1],
                            color='green', linestyle='-', linewidth=2.0, alpha=0.8, zorder=2
                        )
                        self._nominal_traj_lines.append(line)
                    
                    # Draw backup trajectory (orange or red)
                    color = 'orange' if self.controller_type == 'gatekeeper' else 'red'
                    if nominal_len < len(traj):
                        line, = self.ax.plot(
                            traj[nominal_len:, 0], traj[nominal_len:, 1],
                            color=color, linestyle='--', linewidth=2.0, alpha=0.8, zorder=2
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
def highway_scenario_main(controller_type='backup_cbf', save_animation=False):
    """
    Run the highway scenario with specified safety controller.
    
    Args:
        controller_type: 'backup_cbf', 'gatekeeper', or 'shielding'
        save_animation: whether to save animation frames
    """
    # Create highway environment
    env = HighwayEnv(width=60.0, height=12.0, num_lanes=3)
    
    # Create highway-specific configuration
    highway_config = HighwayEnvironmentConfig(
        lane_centers=env.lane_centers,
        target_lane_idx=2  # Target rightmost lane
    )

    # Setup plotting
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Highway Scenario with {controller_type.upper()} Controller")

    # Robot specification
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'radius': 1.0,
        'v_nominal': 2.0,
        'visualize_backup_set': True,
    }

    # Initial state [x, y, θ, v]
    x0 = np.array([-2.0, env.lane_centers[0], 0.0, robot_spec['v_nominal']]).reshape(-1, 1)

    # Controller-specific parameters
    if controller_type == 'backup_cbf':
        nominal_horizon = 2.0  # Not used by backup_cbf
        backup_horizon = 10.0  # Backup CBF horizon
        event_offset = 0.5     # Not used by backup_cbf
    elif controller_type == 'gatekeeper':
        nominal_horizon = 2.0  # Nominal trajectory duration
        backup_horizon = 7.0   # Backup trajectory duration (from end of nominal)
        event_offset = 0.5     # Replanning frequency
    elif controller_type == 'shielding':
        nominal_horizon = 2.0  # Maximum nominal trajectory duration to search
        backup_horizon = 4.0   # Backup trajectory duration (from end of nominal)
        event_offset = 0.5     # Replanning frequency
    else:
        nominal_horizon = 2.0
        backup_horizon = 4.0
        event_offset = 0.5

    # Create controller
    controller = HighwayController(
        X0=x0,
        robot_spec=robot_spec,
        highway_env=env,
        highway_config=highway_config,
        controller_type=controller_type,
        nominal_horizon=nominal_horizon,
        backup_horizon=backup_horizon,
        event_offset=event_offset,
        dt=0.05,
        show_animation=True,
        save_animation=save_animation,
        ax=ax,
        fig=fig
    )

    # Set waypoint
    waypoints = np.array([[55.0, env.lane_centers[0], 0.0]])
    controller.set_waypoints(waypoints)

    # Add traffic obstacles
    controller.init_traffic([
        {
            "x": 15.0,
            "y": env.lane_centers[1],
            "radius": 1.0,
            "vx": 0.5  # slower-moving obstacle in lane 2
        }
        # {
        #     "x": 10.0,
        #     "y": env.lane_centers[0],
        #     "radius": 1.0,
        #     "vx": 1.0  # slower obstacle in same lane as ego (lane 1)
        # }
    ])

    # Run simulation
    print(f"Starting highway scenario simulation with {controller_type.upper()}...")
    print(f"Nominal horizon: {nominal_horizon}s")
    print(f"Backup horizon: {backup_horizon}s")
    if controller_type in ['gatekeeper', 'shielding']:
        print(f"Event offset: {event_offset}s")
    
    try:
        step_count = 0
        while not controller.has_reached_goal():
            controller.control_step()
            controller.draw_plot()
            step_count += 1

            if controller.robot.X[0, 0] >= 55.0:
                break

            # Safety limit
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
    import sys
    
    # Parse command line argument for controller type
    controller_type = 'backup_cbf'  # default
    if len(sys.argv) > 1:
        controller_type = sys.argv[1].lower()
        if controller_type not in ['backup_cbf', 'gatekeeper', 'shielding']:
            print(f"Unknown controller type: {controller_type}")
            print("Valid options: 'backup_cbf', 'gatekeeper', 'shielding'")
            sys.exit(1)
    
    print(f"Running highway scenario with controller: {controller_type}")
    highway_scenario_main(controller_type=controller_type, save_animation=False)
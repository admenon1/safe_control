import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from robots.robot import BaseRobot
from backup_cbf.backup_cbf_qp import BackupCBFQP

"""
Created on November 15th, 2025
@author: Aswin D Menon
============================================
Highway scenario using Backup CBF with lane-change backup controller.
All environment-specific parameters are defined here and given to the BackupCBFQP framework.
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
        self.backup_time = 10.0
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
    def h_safety(self, x, obs, robot_radius, t=None):
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
                 show_animation=True, save_animation=False, ax=None, fig=None):
        self.highway_env = highway_env
        self.highway_config = highway_config
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

        self._backup_traj_lines = []  # ← Store line handles
        self._last_traj_count = 0     # ← Track trajectory count

        # Backup CBF-QP filter (generic framework)
        self.backup_cbf_filter = BackupCBFQP(self.robot, self.robot_spec)
        
        # Inject highway-specific parameters and functions
        self._configure_backup_cbf()

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
        self.backup_cbf_filter.set_environment_callbacks(
            backup_time=self.highway_config.backup_time,
            backup_control_fn=backup_control_fn,
            h_safety_fn=h_safety_fn,
            grad_h_safety_fn=grad_h_safety_fn,
            h_backup_fn=h_backup_fn,
            grad_h_backup_fn=grad_h_backup_fn,
            alpha_fn=alpha_fn,
            alpha_b_fn=alpha_b_fn
        )

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

        # Nominal control
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
        u_safe, intervening = self.backup_cbf_filter.backup_cbf_qp(
            self.robot.X.flatten(), u_des, guard
        )

        # Step robot dynamics
        self.robot.step(u_safe.reshape(-1, 1))

        if self.show_animation:
            self.robot.render_plot()

        return 0 if not intervening else 1

    def draw_plot(self, pause=0.01):
        """Update visualization."""
        if not self.show_animation:
            return

        # ============================================================
        # Clear and redraw lane patches (these change every frame)
        # ============================================================
        for patch in self._lane_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._lane_patches.clear()

        # Render lanes (follow ego vehicle)
        ego_x = float(self.robot.get_position()[0])
        lane_patches = self.highway_env.render_lanes(self.ax, ego_x=ego_x)
        if lane_patches:
            self._lane_patches.extend(lane_patches)

        # ============================================================
        # Only redraw NEW backup trajectories
        # ============================================================
        if hasattr(self.backup_cbf_filter, 'visualize_backup') and self.backup_cbf_filter.visualize_backup:
            trajs = self.backup_cbf_filter.get_backup_trajectories()
            current_traj_count = len(trajs)

            # Only update if NEW trajectories were added
            if current_traj_count > self._last_traj_count:
                # Draw only the NEW trajectories (not all of them!)
                new_trajs = trajs[self._last_traj_count:]
                
                for phi in new_trajs:
                    line, = self.ax.plot(
                        phi[:, 0], phi[:, 1],
                        color='orange', linestyle='--', 
                        linewidth=1.0, alpha=0.7, zorder=2
                    )
                    self._backup_traj_lines.append(line)  # Store handle
                
                self._last_traj_count = current_traj_count

        # Refresh plot (only changed elements)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(pause)
    
    def clear_backup_trajectories(self):
        """Optional: Clear old backup trajectory lines."""
        for line in self._backup_traj_lines:
            try:
                line.remove()
            except:
                pass
        self._backup_traj_lines.clear()
        self._last_traj_count = 0


# ========================================================================
# Main Simulation Entry Point
# ========================================================================
def highway_scenario_main(save_animation=False):
    """Run the highway scenario with backup CBF control."""
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

    # Robot specification
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'v_max': 3.0,
        'radius': 1.0,
        'v_nominal': 2.0,
        'visualize_backup_set': True,
    }

    # Initial state [x, y, θ, v]
    x0 = np.array([-2.0, env.lane_centers[0], 0.0, robot_spec['v_nominal']]).reshape(-1, 1)

    # Create controller
    controller = HighwayController(
        X0=x0,
        robot_spec=robot_spec,
        highway_env=env,
        highway_config=highway_config,  # Pass environment config
        dt=0.05,
        show_animation=True,
        save_animation=save_animation,
        ax=ax,
        fig=fig
    )

    # Set waypoint
    waypoints = np.array([[55.0, env.lane_centers[0], 0.0]])
    controller.set_waypoints(waypoints)

    # Add traffic obstacle
    controller.init_traffic([
        {
            "x": 15.0,
            "y": env.lane_centers[1],
            "radius": 1.0,
            "vx": 0.5
        }
    ])

    # Run simulation
    print("Starting highway scenario simulation...")
    try:
        while not controller.has_reached_goal():
            controller.control_step()
            controller.draw_plot()

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
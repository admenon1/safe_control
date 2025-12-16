import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
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
        self.backup_time = 10.0 # Match backup horizon for responsive lane changes
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
        T = max(self.backup_time, 1e-3) ### 10 does not work for gatekeeper. 4 does not work for backup-CBF
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
        self._last_traj_count = 0     # ← Track trajectory count
        self.obs = np.empty((0, 7))
       
        # Video recording
        self.frames = []  # Store frames for video export

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

    def save_video(self, filename='highway_simulation.mp4', fps=20):
        """Save collected frames as video."""
        if not self.frames:
            print("No frames to save. Run simulation with save_animation=True.")
            return
        
        print(f"Saving video to {filename}...")
        
        # Create animation from frames
        fig_temp = plt.figure(figsize=(12, 6))
        ax_temp = fig_temp.add_subplot(111)
        ax_temp.axis('off')
        
        im = ax_temp.imshow(self.frames[0])
        
        def update_frame(frame_idx):
            im.set_array(self.frames[frame_idx])
            return [im]
        
        anim = animation.FuncAnimation(
            fig_temp, update_frame, frames=len(self.frames),
            interval=1000/fps, blit=True
        )
        
        # Save using ffmpeg writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Highway Simulation'), bitrate=1800)
        anim.save(filename, writer=writer)
        
        plt.close(fig_temp)
        print(f"Video saved successfully: {filename}")
        print(f"Total frames: {len(self.frames)}, Duration: {len(self.frames)/fps:.2f}s")
    
    def draw_plot(self, pause=0.01):
        """Update visualization with incremental trajectory drawing for efficiency."""
        if not self.show_animation:
            return

        # Clear lane patches (need to redraw as view follows ego)
        for patch in self._lane_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self._lane_patches.clear()

        # Clear nominal trajectory lines (committed trajectory changes every frame)
        for line in self._nominal_traj_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._nominal_traj_lines.clear()

        # ============================================================
        # Draw backup trajectories INCREMENTALLY (only new ones)
        # This is the main efficiency optimization
        # ============================================================
        if hasattr(self.safety_controller, 'visualize_backup') and self.safety_controller.visualize_backup:
            trajs = self.safety_controller.get_backup_trajectories()
            current_traj_count = len(trajs)
            
            # Different colors for different controllers
            color = 'orange' if self.controller_type in ['backup_cbf', 'gatekeeper'] else 'red'

            # Only draw NEW trajectories (not already drawn ones)
            if current_traj_count > self._last_traj_count:
                new_trajs = trajs[self._last_traj_count:]
                
                for traj in new_trajs:
                    line, = self.ax.plot(
                        traj[:, 0], traj[:, 1],
                        color=color, linestyle='--', linewidth=1.0, alpha=0.7, zorder=2
                    )
                    self._backup_traj_lines.append(line)
                
                self._last_traj_count = current_traj_count

        # ============================================================
        # Draw committed trajectory for gatekeeper/shielding
        # This MUST be redrawn every frame (it changes as robot moves)
        # ============================================================
        if self.controller_type in ['gatekeeper', 'shielding']:
            if hasattr(self.safety_controller, 'committed_x_traj') and self.safety_controller.committed_x_traj is not None:
                traj = self.safety_controller.committed_x_traj
                
                # Separate nominal and backup portions
                if hasattr(self.safety_controller, 'committed_horizon'):
                    nominal_len = int(self.safety_controller.committed_horizon / self.dt)
                    
                    # Draw nominal trajectory (green) - current plan
                    if nominal_len > 0 and nominal_len <= len(traj):
                        line, = self.ax.plot(
                            traj[:nominal_len, 0], traj[:nominal_len, 1],
                            color='green', linestyle='-', linewidth=2.0, alpha=0.8, zorder=2
                        )
                        self._nominal_traj_lines.append(line)
                    
                    # Draw backup portion of committed trajectory (orange or red)
                    color = 'orange' if self.controller_type == 'gatekeeper' else 'red'
                    if nominal_len < len(traj):
                        line, = self.ax.plot(
                            traj[nominal_len:, 0], traj[nominal_len:, 1],
                            color=color, linestyle='--', linewidth=2.0, alpha=0.8, zorder=2
                        )
                        self._nominal_traj_lines.append(line)

        # Render lanes (redrawn each frame as view follows ego)
        ego_x = float(self.robot.get_position()[0])
        lane_patches = self.highway_env.render_lanes(self.ax, ego_x=ego_x)
        if lane_patches:
            self._lane_patches.extend(lane_patches)

        # Refresh plot efficiently
        if self.save_animation:
            # Full redraw needed for video capture
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            # Faster: only process events, blit handles the rest
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        
        # Capture frame for video if saving animation (after all drawing is complete)
        if self.save_animation:
            # Convert canvas to image array
            buf = self.fig.canvas.buffer_rgba()
            img = np.asarray(buf)
            # Convert RGBA to RGB
            img = img[:, :, :3].copy()  # Make a copy to avoid reference issues
            self.frames.append(img)
        
        plt.pause(pause)

    def clear_backup_trajectories(self):
        """Clear old backup trajectory lines and reset counter."""
        for line in self._backup_traj_lines:
            try:
                line.remove()
            except:
                pass
        self._backup_traj_lines.clear()
        self._last_traj_count = 0
        self._last_traj_count = 0


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
        target_lane_idx=0  # Target bottom lane (US: fast lane on left/bottom)
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
        'v_max': 3.0,
        'radius': 1.0,
        'v_nominal': 2.0,
        'visualize_backup_set': True,
    }

    # Initial state [x, y, θ, v]
    # US system: Start in top lane (lane 2 = slow lane), overtake to bottom lane (lane 0 = fast lane)
    x0 = np.array([-2.0, env.lane_centers[2], 0.0, robot_spec['v_nominal']]).reshape(-1, 1)

    # Controller-specific parameters
    if controller_type == 'backup_cbf':
        nominal_horizon = 2.0  # Not used by backup_cbf
        backup_horizon = 10.0  # Backup CBF horizon
        event_offset = 0.5     # Not used by backup_cbf
    elif controller_type == 'gatekeeper':
        nominal_horizon = 3  # Nominal trajectory duration
        backup_horizon = 3   # Backup trajectory duration (from end of nominal)
        event_offset = 0.5     # Replanning frequency
    elif controller_type == 'shielding':
        nominal_horizon = 3  # Maximum nominal trajectory duration to search
        backup_horizon = 3   # Backup trajectory duration (from end of nominal)
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

    # Set waypoint (US: overtake to bottom/fast lane)
    waypoints = np.array([[40.0, env.lane_centers[0], 0.0]])
    controller.set_waypoints(waypoints)

    # Add traffic obstacles
    # US system: slower traffic in middle lane, ego overtakes from top to bottom
    controller.init_traffic([
        {
            "x": 15.0,
            "y": env.lane_centers[1],  # Middle lane
            "radius": 1.0,
            "vx": 0.5  # Slower-moving obstacle
        }
        # {
        #     "x": 10.0,
        #     "y": env.lane_centers[2],  # Top lane (same as ego start)
        #     "radius": 1.0,
        #     "vx": 1.1  # Slower obstacle in same lane as ego
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
        
        # Save video if requested
        if save_animation and controller.frames:
            video_filename = f"highway_{controller_type}_{step_count}steps.mp4"
            controller.save_video(video_filename, fps=20)
        
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    controller_type = 'backup_cbf'  # default
    save_video = False
    
    # Parse arguments
    for arg in sys.argv[1:]:
        if arg.lower() in ['backup_cbf', 'gatekeeper', 'shielding']:
            controller_type = arg.lower()
        elif arg.lower() in ['--save-video', '-s', '--video']:
            save_video = True
        elif arg.lower() in ['--help', '-h']:
            print("Usage: python highway_main.py [controller_type] [--save-video]")
            print("\nController types:")
            print("  backup_cbf  - Backup CBF-QP controller (default)")
            print("  gatekeeper  - Gatekeeper algorithm")
            print("  shielding   - Shielding algorithm")
            print("\nOptions:")
            print("  --save-video, -s  Save simulation as MP4 video")
            print("\nExample:")
            print("  python highway_main.py shielding --save-video")
            sys.exit(0)
    
    if controller_type not in ['backup_cbf', 'gatekeeper', 'shielding']:
        print(f"Unknown controller type: {controller_type}")
        print("Valid options: 'backup_cbf', 'gatekeeper', 'shielding'")
        sys.exit(1)
    
    print(f"Running highway scenario with controller: {controller_type}")
    if save_video:
        print("Video recording enabled")
    
    highway_scenario_main(controller_type=controller_type, save_animation=save_video)
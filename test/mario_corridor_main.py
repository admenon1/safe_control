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
Created on November 22nd, 2025
@author: Aswin D Menon
Updated on November 25th, 2025
============================================
Modular corridor ball-passing scenario supporting multiple safety controllers:
- backup_cbf 
- gatekeeper
- shielding

Robot moves through corridor while avoiding a continuously moving ball.
Backup plan: Retreat to safety pocket above corridor and stop.

Usage:
    python mario_corridor_main.py [controller_type] [--save-video]
    controller_type: 'backup_cbf' (default), 'gatekeeper', or 'shielding'
"""


# ========================================================================
# Corridor Environment Parameters and Functions
# ========================================================================
class CorridorEnvironmentConfig:
    """
    Encapsulates all corridor-scenario-specific parameters and CBF definitions.
    """
    def __init__(self, corridor_center_y, corridor_height, pocket_center, pocket_size):
        # Corridor parameters
        self.corridor_center_y = corridor_center_y
        self.corridor_height = corridor_height
        self.corridor_bottom = corridor_center_y - corridor_height / 2.0
        self.corridor_top = corridor_center_y + corridor_height / 2.0
        
        # Pocket (safe zone) parameters
        self.pocket_center = np.array(pocket_center)  # [x, y]
        self.pocket_width = pocket_size[0]
        self.pocket_height = pocket_size[1]
        self.pocket_margin = 0.3  # Safety margin inside pocket
        
        # Pocket boundaries
        self.pocket_left = pocket_center[0] - pocket_size[0] / 2.0
        self.pocket_right = pocket_center[0] + pocket_size[0] / 2.0
        self.pocket_bottom = pocket_center[1] - pocket_size[1] / 2.0
        self.pocket_top = pocket_center[1] + pocket_size[1] / 2.0
        
        # KEY POINT: Centerline position directly below pocket
        self.pivot_point = np.array([pocket_center[0], corridor_center_y])

        # Backup controller parameters
        self.backup_time = 10.0
        self.backup_zeta = 0.9
        self.omega_max_backup = 2.0  # VERY AGGRESSIVE turning
        self.target_speed = 0.0
        
        # Wall safety buffer
        self.wall_buffer = 0.5

        # CBF class-K function parameters
        self.alpha_linear_coeff = 20.0
        self.alpha_b_coeff = 10.0

    # ------------------------------------------------------------------
    # Backup Controller (Fixed Two-Line Path)
    # ------------------------------------------------------------------
    def backup_control(self, x, robot_spec):
        """
        Two-line backup controller with guaranteed convergence.
        
        Line 1: Current → Pivot (on centerline below pocket)
        Line 2: Pivot → Pocket center
        
        Args:
            x: state [px, py, theta, v]
            robot_spec: robot specification dict
        
        Returns:
            u_b: backup control [[accel], [omega]]
        """
        x = x.reshape(-1)
        px, py, theta, v = x[:4]

        # ============================================================
        # Check if already in pocket (STOP condition)
        # ============================================================
        dx = px - self.pocket_center[0]
        dy = py - self.pocket_center[1]
        
        a = (self.pocket_width / 2.0) - self.pocket_margin
        b = (self.pocket_height / 2.0) - self.pocket_margin
        in_pocket = (dx / a)**2 + (dy / b)**2 < 1.0
        
        if in_pocket:
            # Inside pocket: STOP
            accel = -5.0 * v
            omega = 0.0
            return np.array([[accel], [omega]])

        # ============================================================
        # Determine target: Pivot or Pocket Center
        # ============================================================
        
        # Distance to pivot point
        dx_pivot = self.pivot_point[0] - px
        dy_pivot = self.pivot_point[1] - py
        dist_to_pivot = np.sqrt(dx_pivot**2 + dy_pivot**2)
        
        # Check if robot is near/above corridor top
        near_pivot = dist_to_pivot < 1.0 or py > (self.corridor_top - 0.5)
        
        if near_pivot:
            # LINE 2: Go to pocket center
            target_x = self.pocket_center[0]
            target_y = self.pocket_center[1]
        else:
            # LINE 1: Go to pivot point
            target_x = self.pivot_point[0]
            target_y = self.pivot_point[1]

        # ============================================================
        # Steer toward target
        # ============================================================
        
        dx_target = target_x - px
        dy_target = target_y - py
        distance_to_target = np.sqrt(dx_target**2 + dy_target**2)
        
        # Desired heading
        theta_desired = np.arctan2(dy_target, dx_target)
        
        # Heading error
        e_psi = theta - theta_desired
        e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))
        
        # Decide forward or backward
        should_reverse = abs(e_psi) > np.pi / 2
        
        if should_reverse:
            # Flip desired heading for reverse motion
            theta_desired += np.pi
            theta_desired = np.arctan2(np.sin(theta_desired), np.cos(theta_desired))
            e_psi = theta - theta_desired
            e_psi = np.arctan2(np.sin(e_psi), np.cos(e_psi))
        
        # Angular velocity
        k_omega = 4.0
        omega = -k_omega * e_psi
        omega = np.clip(omega, -self.omega_max_backup, self.omega_max_backup)
        
        # ============================================================
        # Velocity control
        # ============================================================
        
        if distance_to_target < 0.5:
            v_mag = 0.3  # Slow near target
        elif abs(e_psi) > np.pi / 4:
            v_mag = 0.5  # Slow when turning
        else:
            v_mag = 1.2  # Normal speed
        
        v_desired = -v_mag if should_reverse else v_mag
        
        # Acceleration
        k_accel = 4.0
        accel = k_accel * (v_desired - v)
        accel = np.clip(accel, -robot_spec.get('a_max', 0.5), 
                              robot_spec.get('a_max', 0.5))
        
        return np.array([[accel], [omega]])

    # ------------------------------------------------------------------
    # Safety CBF (Ball Collision Only)
    # ------------------------------------------------------------------
    def h_safety(self, x, ball, robot_radius):
        """
        Simple safety CBF: Ball collision avoidance only.
        
        Args:
            x: state [px, py, theta, v]
            ball: ball state [bx, by, radius, vx, vy, ...]
            robot_radius: ego robot radius
        
        Returns:
            h: CBF value (positive = safe)
        """
        px, py = x[:2]
        
        # Ball collision avoidance
        bx, by, ball_r = ball[:3]
        combined_radius = ball_r + robot_radius
        h_ball = (px - bx) ** 2 + (py - by) ** 2 - combined_radius ** 2
        
        return h_ball

    def grad_h_safety(self, x, ball):
        """
        Gradient of safety CBF (ball collision only).
        
        Returns:
            grad_h: gradient vector (nx,)
        """
        px, py = x[:2]
        bx, by = ball[:2]
        
        # Gradient for ball collision
        grad = np.zeros(x.size)
        grad[0] = 2.0 * (px - bx)
        grad[1] = 2.0 * (py - by)
        
        return grad

    # ------------------------------------------------------------------
    # Terminal CBF (Inside Pocket and Stopped)
    # ------------------------------------------------------------------
    def h_backup(self, x):
        """
        Terminal CBF: Robot is inside pocket and stopped.
        
        Returns:
            h_b: CBF value (positive = inside pocket and stopped)
        """
        px, py, _, v = x[:4]
        
        # Distance from pocket center
        dx = px - self.pocket_center[0]
        dy = py - self.pocket_center[1]
        
        # Ellipse constraint: (dx/a)² + (dy/b)² < 1
        a = (self.pocket_width / 2.0) - self.pocket_margin
        b = (self.pocket_height / 2.0) - self.pocket_margin
        
        spatial_term = 1.0 - (dx / a)**2 - (dy / b)**2
        velocity_term = -(v - self.target_speed)**2 + 0.1  # Allow v ≈ 0
        
        # Combined: both must be positive
        h_b = min(spatial_term, velocity_term)
        return h_b

    def grad_h_backup(self, x):
        """
        Gradient of terminal CBF with respect to state.
        
        Returns:
            grad_h_b: gradient vector (nx,)
        """
        px, py, _, v = x[:4]
        dx = px - self.pocket_center[0]
        dy = py - self.pocket_center[1]
        
        a = (self.pocket_width / 2.0) - self.pocket_margin
        b = (self.pocket_height / 2.0) - self.pocket_margin
        
        grad = np.zeros(x.size)
        grad[0] = -2.0 * dx / (a**2)
        grad[1] = -2.0 * dy / (b**2)
        grad[3] = -2.0 * (v - self.target_speed)
        
        return grad

    # ------------------------------------------------------------------
    # Class-K Functions (CBF Enforcement)
    # ------------------------------------------------------------------
    def alpha(self, h):
        """Class-K function for safety CBF."""
        return self.alpha_linear_coeff * h

    def alpha_b(self, h_b):
        """Class-K function for terminal CBF."""
        return self.alpha_b_coeff * h_b


# ========================================================================
# Corridor Environment (Visualization)
# ========================================================================
class CorridorEnv:
    def __init__(self, width=30.0, corridor_height=3.0, corridor_center_y=1.5,
                 pocket_center=(5.0, 4.5), pocket_size=(4.0, 2.0)):
        self.width = width
        self.corridor_height = corridor_height
        self.corridor_center_y = corridor_center_y
        self.corridor_bottom = corridor_center_y - corridor_height / 2.0
        self.corridor_top = corridor_center_y + corridor_height / 2.0
        
        self.pocket_center = pocket_center
        self.pocket_width = pocket_size[0]
        self.pocket_height = pocket_size[1]

    def render_environment(self, ax):
        """Draw corridor boundaries and safety pocket."""
        # Corridor boundaries (fixed view)
        # Bottom wall
        ax.plot([0, self.width], 
                [self.corridor_bottom, self.corridor_bottom],
                color='black', linewidth=2.0, zorder=0)
        
        # Top wall (stops at pocket entrance)
        pocket_left_x = self.pocket_center[0] - self.pocket_width / 2
        pocket_right_x = self.pocket_center[0] + self.pocket_width / 2
        
        ax.plot([0, pocket_left_x], 
                [self.corridor_top, self.corridor_top],
                color='black', linewidth=2.0, zorder=0)
        ax.plot([pocket_right_x, self.width], 
                [self.corridor_top, self.corridor_top],
                color='black', linewidth=2.0, zorder=0)
        
        # Centerline
        ax.plot([0, self.width], 
                [self.corridor_center_y, self.corridor_center_y],
                color='gray', linestyle='--', linewidth=1.0, zorder=0)
        
        # Safety pocket (rectangle above corridor)
        pocket_rect = patches.Rectangle(
            (self.pocket_center[0] - self.pocket_width/2, 
             self.pocket_center[1] - self.pocket_height/2),
            self.pocket_width, self.pocket_height,
            linewidth=2, edgecolor='green', facecolor='lightgreen',
            alpha=0.3, zorder=1
        )
        ax.add_patch(pocket_rect)
        
        # Draw pocket walls
        pocket_left = self.pocket_center[0] - self.pocket_width/2
        pocket_right = self.pocket_center[0] + self.pocket_width/2
        pocket_bottom = self.pocket_center[1] - self.pocket_height/2
        pocket_top = self.pocket_center[1] + self.pocket_height/2
        
        # Left wall of pocket
        ax.plot([pocket_left, pocket_left], 
                [pocket_bottom, pocket_top],
                color='green', linewidth=2.0, zorder=1)
        # Right wall of pocket
        ax.plot([pocket_right, pocket_right], 
                [pocket_bottom, pocket_top],
                color='green', linewidth=2.0, zorder=1)
        # Top wall of pocket
        ax.plot([pocket_left, pocket_right], 
                [pocket_top, pocket_top],
                color='green', linewidth=2.0, zorder=1)


# ========================================================================
# Corridor Controller
# ========================================================================
class CorridorController:
    def __init__(self, X0, robot_spec, corridor_env, corridor_config, 
                 ball_params, dt=0.05, 
                 controller_type='backup_cbf', nominal_horizon=2.0, backup_horizon=4.0, event_offset=0.5,
                 show_animation=True, save_animation=False, ax=None, fig=None):
        self.corridor_env = corridor_env
        self.corridor_config = corridor_config
        self.dt = dt
        self.controller_type = controller_type
        self.show_animation = show_animation
        self.save_animation = save_animation
        self.ax = ax or plt.axes()
        self.fig = fig or plt.figure()

        # Robot setup
        self.robot_spec = robot_spec
        self.robot = BaseRobot(X0.reshape(-1, 1), robot_spec, dt, self.ax)

        # Ball parameters
        self.ball_params = ball_params
        self.ball_state = np.array([
            ball_params['x0'], ball_params['y0'], ball_params['radius'],
            ball_params['vx'], 0.0, 0.0, 0.0
        ])
        self.ball_patch = None
        
        # Video recording
        self.frames = []  # Store frames for video export
        
        # Visualization tracking
        self._backup_traj_lines = []
        self._nominal_traj_lines = []

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

        # Goal
        self.goal_x = None

        # Override robot's nominal controller
        self.robot.nominal_input = self.nominal_input_corridor

    def _configure_backup_cbf(self):
        """Inject corridor-specific callbacks into BackupCBFQP."""
        backup_control_fn = lambda x: self.corridor_config.backup_control(x, self.robot_spec)
        h_safety_fn = self.corridor_config.h_safety
        grad_h_safety_fn = self.corridor_config.grad_h_safety
        h_backup_fn = self.corridor_config.h_backup
        grad_h_backup_fn = self.corridor_config.grad_h_backup
        alpha_fn = self.corridor_config.alpha
        alpha_b_fn = self.corridor_config.alpha_b

        self.safety_controller.set_environment_callbacks(
            backup_time=self.corridor_config.backup_time,
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
        return self.nominal_input_corridor(target_speed=self.robot_spec.get('v_nominal', 1.5))
    
    def backup_controller_wrapper(self, state):
        """Wrapper for Gatekeeper/Shielding to call backup controller."""
        return self.corridor_config.backup_control(state, self.robot_spec)

    def nominal_input_corridor(self, target_speed=1.5, **kwargs):
        """
        Nominal controller: Follow corridor centerline at constant speed.
        """
        X = self.robot.X
        py = X[1, 0]
        v = X[3, 0]
        theta = X[2, 0]
        
        # Controller gains
        k_a = 1.0
        k_y = 2.0
        target_heading = 0.0  # Drive straight (horizontal)
        
        # Speed regulation
        accel = k_a * (target_speed - v)
        
        # Lateral centering + heading alignment
        y_error = py - self.corridor_env.corridor_center_y
        omega = -k_y * y_error - 0.5 * theta
        
        # Clip to limits
        accel = np.clip(accel, -self.robot_spec['a_max'], self.robot_spec['a_max'])
        omega = np.clip(omega, -self.robot_spec['w_max'], self.robot_spec['w_max'])
        
        return np.array([[accel], [omega]])

    def init_ball(self):
        """Initialize ball visualization."""
        self.ball_patch = self.ax.add_patch(
            patches.Circle(
                (self.ball_state[0], self.ball_state[1]),
                self.ball_state[2],
                facecolor='red',
                edgecolor='darkred',
                alpha=0.8,
                zorder=4
            )
        )

    def update_ball(self):
        """Update ball position with wrapping."""
        self.ball_state[0] += self.ball_state[3] * self.dt
        
        # Wrap around when ball exits right side
        if self.ball_state[0] > self.ball_params['x_max']:
            self.ball_state[0] = self.ball_params['x_min']
        
        # Update visualization
        if self.ball_patch is not None:
            self.ball_patch.center = (self.ball_state[0], self.ball_state[1])

    def set_goal(self, goal_x):
        """Set goal x-position."""
        self.goal_x = goal_x
        
        # Initialize control reference for gatekeeper/shielding
        if self.controller_type in ['gatekeeper', 'shielding']:
            goal = np.array([goal_x, self.corridor_env.corridor_center_y])
            self.control_ref = {
                'goal': goal.reshape(-1, 1),
                'state_machine': 'track',
                'u_ref': np.zeros((2, 1))
            }

    def has_reached_goal(self):
        """Check if robot reached goal."""
        if self.goal_x is None:
            return False
        return self.robot.X[0, 0] >= self.goal_x

    def control_step(self):
        """Execute one control loop iteration."""
        self.update_ball()

        if self.controller_type == 'backup_cbf':
            # Backup CBF-QP mode
            u_des = self.nominal_input_corridor(
                target_speed=self.robot_spec.get('v_nominal', 1.5)
            ).flatten()

            # Apply backup CBF filter (ball is the obstacle)
            u_safe, intervening = self.safety_controller.backup_cbf_qp(
                self.robot.X.flatten(), u_des, self.ball_state
            )
            
            # Step robot dynamics
            self.robot.step(u_safe.reshape(-1, 1))
            return_val = 0 if not intervening else 1
            
        elif self.controller_type in ['gatekeeper', 'shielding']:
            # Gatekeeper/Shielding mode
            # Ball as obstacle array
            obs = self.ball_state.reshape(1, -1)
            
            u_safe = self.safety_controller.solve_control_problem(
                self.robot.X,
                self.control_ref,
                obs
            )
            
            # Step robot dynamics
            self.robot.step(u_safe.reshape(-1, 1))
            return_val = 0

        if self.show_animation:
            self.robot.render_plot()

        return return_val

    def save_video(self, filename='corridor_simulation.mp4', fps=20):
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
        writer = Writer(fps=fps, metadata=dict(artist='Corridor Simulation'), bitrate=1800)
        anim.save(filename, writer=writer)
        
        plt.close(fig_temp)
        print(f"Video saved successfully: {filename}")
        print(f"Total frames: {len(self.frames)}, Duration: {len(self.frames)/fps:.2f}s")

    def draw_plot(self, pause=0.01):
        """Update visualization (static view)."""
        if not self.show_animation:
            return

        # Clear previous trajectory lines
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

        # Refresh plot
        self.fig.canvas.draw()
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


# ========================================================================
# Main Simulation Entry Point
# ========================================================================
def corridor_scenario_main(controller_type='backup_cbf', save_animation=False):
    """Run the corridor ball-passing scenario with specified safety controller.
    
    Args:
        controller_type: 'backup_cbf', 'gatekeeper', or 'shielding'
        save_animation: whether to save animation frames
    """
    
    # Environment parameters (shorter corridor)
    corridor_width = 20.0
    corridor_height = 3.0
    corridor_center_y = 1.5
    pocket_center = (10.0, 4.5)  # Above corridor
    pocket_size = (6.0, 3.0)    # Width × Height

    # Create environment
    env = CorridorEnv(
        width=corridor_width,
        corridor_height=corridor_height,
        corridor_center_y=corridor_center_y,
        pocket_center=pocket_center,
        pocket_size=pocket_size
    )

    # Create corridor-specific configuration
    corridor_config = CorridorEnvironmentConfig(
        corridor_center_y=corridor_center_y,
        corridor_height=corridor_height,
        pocket_center=pocket_center,
        pocket_size=pocket_size
    )

    # Ball parameters (moves horizontally through corridor)
    ball_params = {
        'x0': -5.0,          # Start position
        'y0': corridor_center_y,  # Corridor centerline
        'radius': 1.0,
        'vx': 1.0,           # Faster than robot
        'x_min': -5.0,       # Wrap position
        'x_max': corridor_width + 5.0  # Wrap position
    }

    # Setup plotting (static view)
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes()
    ax.set_aspect('equal')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Corridor Scenario with {controller_type.upper()} Controller")
    ax.set_xlim(-2, corridor_width + 2)  # Fixed view
    ax.set_ylim(-0.5, 6.5)

    # Draw environment once
    env.render_environment(ax)

    # Robot specification
    robot_spec = {
        'model': 'DynamicUnicycle2D',
        'w_max': 0.5,
        'a_max': 0.5,
        'radius': 0.3,
        'v_nominal': 0.5,  # Slower than ball
        'visualize_backup_set': True,
    }

    # Initial state [x, y, θ, v] - start in corridor
    x0 = np.array([5, corridor_center_y, 0.0, robot_spec['v_nominal']]).reshape(-1, 1)

    # Controller-specific parameters
    if controller_type == 'backup_cbf':
        nominal_horizon = 2.0  # Not used by backup_cbf
        backup_horizon = 10.0  # Backup CBF horizon
        event_offset = 0.5     # Not used by backup_cbf
    elif controller_type == 'gatekeeper':
        nominal_horizon = 6  # Nominal trajectory duration
        backup_horizon = 6   # Backup trajectory duration (from end of nominal)
        event_offset = 0.5     # Replanning frequency
    elif controller_type == 'shielding':
        nominal_horizon = 6  # Maximum nominal trajectory duration to search
        backup_horizon = 6   # Backup trajectory duration (from end of nominal)
        event_offset = 0.5     # Replanning frequency
    else:
        nominal_horizon = 2.0
        backup_horizon = 4.0
        event_offset = 0.5

    # Create controller
    controller = CorridorController(
        X0=x0,
        robot_spec=robot_spec,
        corridor_env=env,
        corridor_config=corridor_config,
        ball_params=ball_params,
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

    # Initialize ball
    controller.init_ball()

    # Set goal
    controller.set_goal(goal_x=corridor_width - 3.0)

    # Run simulation
    print(f"Starting corridor ball-passing scenario simulation with {controller_type.upper()}...")
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

            if controller.robot.X[0, 0] >= corridor_width:
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
            video_filename = f"corridor_{controller_type}_{step_count}steps.mp4"
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
            print("Usage: python mario_corridor_main.py [controller_type] [--save-video]")
            print("\nController types:")
            print("  backup_cbf  - Backup CBF-QP controller (default)")
            print("  gatekeeper  - Gatekeeper algorithm")
            print("  shielding   - Shielding algorithm")
            print("\nOptions:")
            print("  --save-video, -s  Save simulation as MP4 video")
            print("\nExample:")
            print("  python mario_corridor_main.py shielding --save-video")
            sys.exit(0)
    
    if controller_type not in ['backup_cbf', 'gatekeeper', 'shielding']:
        print(f"Unknown controller type: {controller_type}")
        print("Valid options: 'backup_cbf', 'gatekeeper', 'shielding'")
        sys.exit(1)
    
    print(f"Running corridor scenario with controller: {controller_type}")
    if save_video:
        print("Video recording enabled")
    
    corridor_scenario_main(controller_type=controller_type, save_animation=save_video)
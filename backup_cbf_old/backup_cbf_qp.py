import numpy as np
import cvxpy as cp

"""
Created on November 15th, 2025
@author: Aswin D Menon
============================================
Backup CBF-QP implementation.
Environment-specific parameters should be provided.
"""


class BackupCBFQP:
    """
    Generic Backup CBF-QP Framework.
    
    - Simulates a backup controller for a fixed horizon.
    - Propagates sensitivity matrices along the backup trajectory.
    - Enforces safety and terminal constraints via QP.
    
    QP Formulation:
        minimize    ||u - u_des||²        [2D control: acceleration, omega]
        subject to  G @ u >= h            [N+1 constraints: N safety + 1 terminal]

    The QP has 2 decision variables and N+1 linear inequality constraints.
    
    Environment-specific components (must be provided via set_environment_callbacks):
    - backup_control_fn: Backup controller u_b(x)
    - h_safety_fn: Safety CBF h(x, obs)
    - grad_h_safety_fn: Gradient ∇h(x, obs)
    - h_backup_fn: Terminal CBF h_b(x)
    - grad_h_backup_fn: Gradient ∇h_b(x)
    - alpha_fn: Class-K function for safety CBF
    - alpha_b_fn: Class-K function for terminal CBF
    """

    def __init__(self, robot, robot_spec):
        self.robot = robot                # robots.robot.BaseRobot instance
        self.robot_spec = robot_spec

        self.nx = self.robot.X.shape[0]
        self.nu = self.robot.U.shape[0]

        # Horizon and integration step
        self.dt = self.robot.dt
        self.backup_time = None           # To be set
        self.N = None                     # To be set

        # Control limits
        self.u_max = self.robot_spec.get("a_max", 1.0)
        self.w_max = self.robot_spec.get("w_max", 0.5)

        # Robot geometric properties
        self.robot_radius = getattr(self.robot, "robot_radius", 0.3)

        # Visualization flag
        self.visualize_backup = self.robot_spec.get("visualize_backup_set", False)
        self.backup_trajs = []
        self.save_every_N = 5
        self.curr_step = 0

        # Environment-specific callbacks (to be set by environment)
        self.backup_control_fn = None
        self.h_safety_fn = None
        self.grad_h_safety_fn = None
        self.h_backup_fn = None
        self.grad_h_backup_fn = None
        self.alpha_fn = None
        self.alpha_b_fn = None

    def set_environment_callbacks(self, 
                                 backup_time,
                                 backup_control_fn,
                                 h_safety_fn,
                                 grad_h_safety_fn,
                                 h_backup_fn,
                                 grad_h_backup_fn,
                                 alpha_fn,
                                 alpha_b_fn):
        """
        Set environment-specific parameters and callback functions.
        
        Args:
            backup_time: Backup controller horizon (seconds)
            backup_control_fn: u_b = f(x) - Returns control (nu,)
            h_safety_fn: h = f(x, obs, robot_radius) - Safety CBF
            grad_h_safety_fn: ∇h = f(x, obs) - Returns gradient (nx,)
            h_backup_fn: h_b = f(x) - Terminal CBF
            grad_h_backup_fn: ∇h_b = f(x) - Returns gradient (nx,)
            alpha_fn: α = f(h) - Class-K function for safety
            alpha_b_fn: α_b = f(h_b) - Class-K function for terminal
        """
        self.backup_time = backup_time
        self.N = int(np.ceil(self.backup_time / self.dt))
        self.backup_control_fn = backup_control_fn
        self.h_safety_fn = h_safety_fn
        self.grad_h_safety_fn = grad_h_safety_fn
        self.h_backup_fn = h_backup_fn
        self.grad_h_backup_fn = grad_h_backup_fn
        self.alpha_fn = alpha_fn
        self.alpha_b_fn = alpha_b_fn

    # ------------------------------------------------------------------
    # Core dynamics helpers
    # ------------------------------------------------------------------
    def f(self, x):
        return self.robot.robot.f(x.reshape(-1, 1)).flatten()

    def g(self, x):
        return self.robot.robot.g(x.reshape(-1, 1))

    def df_dx(self, x):
        if hasattr(self.robot.robot, "df_dx"):
            return self.robot.robot.df_dx(x.reshape(-1, 1))
        return self._finite_difference(self.f, x)

    def dg_dx(self, x):
        if hasattr(self.robot.robot, "dg_dx"):
            return self.robot.robot.dg_dx(x.reshape(-1, 1))
        return np.zeros((self.nu, self.nx, self.nx))

    @staticmethod
    def _finite_difference(func, x, eps=1e-5):
        x = np.array(x, dtype=float)
        base = func(x.copy())
        jac = np.zeros((base.size, x.size))
        for i in range(x.size):
            x_pert = x.copy()
            x_pert[i] += eps
            jac[:, i] = (func(x_pert) - base) / eps
        return jac

    # ------------------------------------------------------------------
    # Combined backup flow for [x; vec(S)]
    # ------------------------------------------------------------------
    def backup_flow(self, z):
        """
        Combined dynamics for state and sensitivity matrix.
        Uses environment-provided backup controller.
        """
        x = z[:self.nx]
        S = z[self.nx:].reshape(self.nx, self.nx)

        f_val = self.f(x).reshape(-1)
        g_val = self.g(x)
        u_b = self.backup_control_fn(x)  # ← Environment-specific controller

        # x_dot
        x_dot = f_val + (g_val @ u_b).reshape(-1)

        # S_dot = (df/dx + dg/dx * u_b) * S
        df = self.df_dx(x)
        dg = self.dg_dx(x)
        A = df.copy()
        for i in range(self.nu):
            A += dg[i] * u_b[i, 0]
        S_dot = (A @ S).reshape(-1)
        return np.concatenate([x_dot, S_dot.flatten()])

    def integrate_backup(self, x0):
        """Integrate x and S over the backup horizon."""
        if self.N is None:
            raise ValueError("Call set_environment_callbacks() first to set backup_time!")
        
        phi = np.zeros((self.N, self.nx))
        S_all = np.zeros((self.N, self.nx, self.nx))

        z = np.concatenate([x0.flatten(), np.eye(self.nx).flatten()])
        phi[0] = x0.flatten()
        S_all[0] = np.eye(self.nx)

        for i in range(1, self.N):
            z = z + self.dt * self.backup_flow(z)
            phi[i] = z[:self.nx]
            S_all[i] = z[self.nx:].reshape(self.nx, self.nx)

        return phi, S_all

    # ------------------------------------------------------------------
    # Main Backup CBF-QP
    # ------------------------------------------------------------------
    def backup_cbf_qp(self, x_curr, u_des, obs_vec):
        """
        Solve the backup CBF-QP to filter nominal control.
        
        Args:
            x_curr: current state (nx,)
            u_des: nominal control (nu,)
            obs_vec: obstacle parameters (environment-specific format)
        
        Returns:
            u_act: safe control (nu,)
            intervening: True if QP modified the control
        """
        if self.backup_control_fn is None:
            raise ValueError("Call set_environment_callbacks() first!")
        
        x_curr = np.array(x_curr, dtype=float)
        u_des = np.array(u_des, dtype=float)
        phi, S_all = self.integrate_backup(x_curr)

        # Store backup trajectories for visualization
        if self.visualize_backup and self.curr_step % self.save_every_N == 0:
            self.backup_trajs.append(phi.copy())
        self.curr_step += 1

        f0 = self.f(x_curr)
        g0 = self.g(x_curr)

        G_list = []
        h_list = []

        # Future safety constraints
        for i in range(1, self.N):
            x_i = phi[i]
            S_i = S_all[i]

            h_val = self.h_safety_fn(x_i, obs_vec, self.robot_radius)  
            grad_h = self.grad_h_safety_fn(x_i, obs_vec)               

            lhs = grad_h @ S_i @ g0
            rhs = -(grad_h @ S_i @ f0 + self.alpha_fn(h_val))  
            G_list.append(lhs)
            h_list.append(rhs)

        # Terminal reachability constraint
        x_T = phi[-1]
        S_T = S_all[-1]

        hb_val = self.h_backup_fn(x_T)              
        grad_hb = self.grad_h_backup_fn(x_T)        

        lhs = grad_hb @ S_T @ g0
        rhs = -(grad_hb @ S_T @ f0 + self.alpha_b_fn(hb_val))  
        G_list.append(lhs)
        h_list.append(rhs)

        G = np.array(G_list)
        h = np.array(h_list)

        # Solve QP using cvxpy
        u = cp.Variable(self.nu)
        objective = cp.Minimize(cp.sum_squares(u - u_des))
        constraints = [G @ u >= h]
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.GUROBI, verbose=False)
            if prob.status == 'optimal':
                u_act = u.value
            else:
                u_act = u_des.copy()
        except Exception:
            u_act = u_des.copy()

        u_act = np.clip(u_act, -self.u_max, self.u_max)
        intervening = np.linalg.norm(u_act - u_des) > 1e-4
        return u_act, intervening
    
    def get_backup_trajectories(self):
        """Return stored backup trajectories for plotting."""
        return self.backup_trajs.copy() if self.visualize_backup else []
    
    def clear_trajectories(self):
        """Clear stored backup trajectories."""
        self.backup_trajs.clear()
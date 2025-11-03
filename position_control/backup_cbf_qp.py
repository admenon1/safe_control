import numpy as np
import quadprog


class BackupASIF:
    """
    Plain (non-robust) backup ASIF.
    - Simulates the backup controller (lane 3 + brake) for a fixed horizon.
    - Propagates the sensitivity matrix for each time slice.
    - Enforces safety and terminal reachability constraints.
    - Filters the nominal control with a 2×2 QP.
    """

    def __init__(self, robot, robot_spec):
        self.robot = robot                # robots.robot.BaseRobot instance
        self.robot_spec = robot_spec

        self.nx = self.robot.X.shape[0]
        self.nu = self.robot.U.shape[0]

        # Horizon and integration step
        self.dt = self.robot.dt
        self.backup_time = 1.5
        self.N = int(np.ceil(self.backup_time / self.dt))

        # Tightening parameters (non-robust)
        self.Lh_const = 1.0
        self.Lhb_const = 1.0
        self.sup_fcl = 6.0
        self.u_max = self.robot_spec.get("a_max", 1.0)

        # Lane-change target
        lane_centers = self.robot_spec.get("lane_centers", [0.0, 3.0, 6.0])
        self.target_lane_idx = self.robot_spec.get("backup_lane_index", 2)
        self.target_lane = lane_centers[self.target_lane_idx]
        self.target_speed = 0.0
        self.lane_margin = 0.25

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
    # Backup controller (lane change)
    # ------------------------------------------------------------------
    def backup_control(self, x):
        x = x.reshape(-1)
        px, py, theta, v = x[:4]

        # Lateral error
        y_err = self.target_lane - py
        k_lat = 1.5
        omega = k_lat * y_err

        # Decelerate to target speed
        accel = 1.5 * (self.target_speed - v)
        return np.array([[accel], [omega]])

    # ------------------------------------------------------------------
    # Combined backup flow for [x; vec(S)]
    # ------------------------------------------------------------------
    def backup_flow(self, z):
        x = z[:self.nx]
        S = z[self.nx:].reshape(self.nx, self.nx)

        f_val = self.f(x).reshape(-1)              # ensure 1-D
        g_val = self.g(x)
        u_b = self.backup_control(x)

        # x_dot
        x_dot = f_val + (g_val @ u_b).reshape(-1)  # 1-D forward dynamics

        # S_dot = (df/dx + dg/dx * u_b) * S
        df = self.df_dx(x)
        dg = self.dg_dx(x)
        A = df.copy()
        for i in range(self.nu):
            A += dg[i] * u_b[i, 0]
        S_dot = (A @ S).reshape(-1)                # flatten sensitivity
        return np.concatenate([x_dot, S_dot.flatten()])

    def integrate_backup(self, x0):
        """
        Integrate x and S over the backup horizon.
        """
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
    # CBF definitions
    # ------------------------------------------------------------------
    def h_safety(self, x, obs, robot_radius):
        px, py = x[:2]
        ox, oy, r = obs[:3]
        combined = r + robot_radius
        return (px - ox) ** 2 + (py - oy) ** 2 - combined ** 2

    def grad_h_safety(self, x, obs):
        px, py = x[:2]
        ox, oy = obs[:2]
        grad = np.zeros(self.nx)
        grad[0] = 2.0 * (px - ox)
        grad[1] = 2.0 * (py - oy)
        return grad

    def h_backup(self, x):
        _, py, _, v = x[:4]
        lane_err = py - self.target_lane
        speed_err = v - self.target_speed
        return -0.5 * (lane_err ** 2 + speed_err ** 2 - self.lane_margin ** 2)

    def grad_h_backup(self, x):
        grad = np.zeros(self.nx)
        grad[1] = -(x[1] - self.target_lane)
        grad[3] = -(x[3] - self.target_speed)
        return grad

    def alpha(self, x):
        return 15.0 * x + x ** 3

    def alpha_b(self, x):
        return 10.0 * x

    # ------------------------------------------------------------------
    # Main ASIF
    # ------------------------------------------------------------------
    def asif(self, x_curr, u_des, obs_vec):
        """
        x_curr: current state (nx,)
        u_des : nominal control (nu,)
        obs_vec: obstacle parameters (at least [x, y, radius, vx, vy])
        returns: u_filtered (nu,), intervening flag
        """
        x_curr = np.array(x_curr, dtype=float)
        u_des = np.array(u_des, dtype=float)
        phi, S_all = self.integrate_backup(x_curr)

        f0 = self.f(x_curr)
        g0 = self.g(x_curr)

        # QP variables
        M = np.eye(self.nu)
        q = u_des.copy()

        G_list = []
        h_list = []

        # Discretization tightening
        mu_d = 0.5 * self.dt * self.Lh_const * self.sup_fcl
        robot_radius = getattr(self.robot, "robot_radius", 0.3)

        # Future safety constraints
        for i in range(1, self.N):
            x_i = phi[i]
            S_i = S_all[i]

            h_val = self.h_safety(x_i, obs_vec, robot_radius)
            grad_h = self.grad_h_safety(x_i, obs_vec)

            lhs = grad_h @ S_i @ g0
            rhs = -(grad_h @ S_i @ f0 + self.alpha(h_val - mu_d))
            G_list.append(lhs)
            h_list.append(rhs)

        # Terminal reachability
        x_T = phi[-1]
        S_T = S_all[-1]

        hb_val = self.h_backup(x_T)
        grad_hb = self.grad_h_backup(x_T)

        lhs = grad_hb @ S_T @ g0
        rhs = -(grad_hb @ S_T @ f0 + self.alpha_b(hb_val))
        G_list.append(lhs)
        h_list.append(rhs)

        G = np.array(G_list)
        h = np.array(h_list)

        try:
            sol = quadprog.solve_qp(M, q, G.T, h, 0)
            u_act = sol[0]
        except Exception:
            u_act = u_des.copy()

        u_act = np.clip(u_act, -self.u_max, self.u_max)
        intervening = np.linalg.norm(u_act - u_des) > 1e-4
        return u_act, intervening
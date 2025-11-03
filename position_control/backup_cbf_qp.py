import numpy as np
import cvxpy as cp

class BackupCBFQP:
    """
    Practical backup-CBF QP for DynamicUnicycle2D.
    Implements T+1 constraints along a simulated backup trajectory and uses
    a backup reference (stop or lane_change) when triggered by a nearby obstacle.
    """

    def __init__(self, robot, robot_spec, num_obs=1, env=None):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs
        self.env = env

        # backup settings (tune)
        self.trigger_distance = robot_spec.get('backup_trigger_dist', 6.0)
        self.backup_type = robot_spec.get('backup_type', 'stop')
        self.lane_centers = robot_spec.get('lane_centers', None)
        self.backup_lane_index = robot_spec.get('backup_lane_index', None)

        # CBF params
        if self.robot_spec['model'] == 'DynamicUnicycle2D':
            self.cbf_param = {'alpha1': 1.5, 'alpha2': 1.5}
        else:
            self.cbf_param = {'alpha': 1.0}

        # qp / problem placeholders
        self.T = 8
        self.u = None
        self.u_ref = None
        self.A = None
        self.b = None
        self.cbf_controller = None
        self.status = 'unknown'

        self.setup_control_problem()

    def setup_control_problem(self):
        """Setup the T+1 constraint QP"""
        # keep T small for interactive demo
        self.T = int(self.T)
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        # T future + 1 current constraints
        self.A = cp.Parameter((self.T + 1, 2), value=np.zeros((self.T + 1, 2)))
        self.b = cp.Parameter((self.T + 1, 1), value=np.zeros((self.T + 1, 1)))

        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
        constraints = [
            self.A @ self.u + self.b >= 0,
            cp.abs(self.u[0]) <= self.robot_spec.get('a_max', 1.0),
            cp.abs(self.u[1]) <= self.robot_spec.get('w_max', 1.0)
        ]

        # fall back solver choice if GUROBI not available
        try:
            self.cbf_controller = cp.Problem(objective, constraints)
        except Exception:
            self.cbf_controller = cp.Problem(objective, constraints)

    def compute_backup_ref(self, X=None):
        """Return (2,1) backup reference control: stop or simple lane-change"""
        state = self.robot.X if X is None else np.array(X, dtype=float).reshape(-1, 1)

        # emergency stop reference (DynamicUnicycle2D: [accel, omega])
        if self.backup_type == 'stop' or self.robot_spec['model'] != 'DynamicUnicycle2D':
            return self.robot.robot.stop(state)
        # lane change reference
        if self.backup_type == 'lane_change':
            lane_centers = self.lane_centers
            if lane_centers is None and self.env is not None and hasattr(self.env, 'lane_centers'):
                lane_centers = self.env.lane_centers
            if lane_centers is None:
                return self.robot.stop(self.robot.X)
            idx = self.backup_lane_index if self.backup_lane_index is not None else (len(lane_centers)-1)
            target_y = lane_centers[idx]

            px, py, theta, v = (float(state[0, 0]), float(state[1, 0]), float(state[2, 0]), float(state[3, 0]))
            forward_x = px + max(1.0, 1.5*v)
            desired_theta = np.arctan2(target_y - py, forward_x - px)
            yaw_err = ((desired_theta - theta + np.pi) % (2*np.pi)) - np.pi
            k_omega = 2.0
            omega = k_omega * yaw_err
            desired_v = 0.2
            k_a = 1.0
            accel = k_a * (desired_v - v)
            return np.array([accel, omega]).reshape(-1, 1)
        return self.robot.robot.stop(state)

    def compute_backup_trajectory(self, X0):
        """
        Simulate backup trajectory under the backup policy starting from X0.
        Returns phi: (T+1, n_x) states array.
        Uses robot.step() wrapper which expects (n,1) arrays.
        """
        n = X0.reshape(-1,1).shape[0]
        phi = np.zeros((self.T + 1, n))
        X_cur = np.array(X0, dtype=float).reshape(-1, 1)
        phi[0, :] = X_cur.flatten()
        for t in range(self.T):
            u_b = self.compute_backup_ref(X_cur)
            X_cur = self.robot.robot.step(X_cur.copy(), u_b)
            phi[t + 1, :] = X_cur.flatten()
        return phi

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        """
        Build T+1 CBF linear constraints and solve QP to return safe control.
        robot_state: current state (n,1)
        control_ref: dict with 'u_ref' (2x1)
        obs_list: list/array of obstacles (may be None)
        """
        # build nominal/ref selection: detect nearest obstacle and trigger backup
        use_backup = False
        nearest_obs = None
        if obs_list is not None and len(obs_list) > 0:
            # ensure shape
            obs_arr = np.array(obs_list)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)
            robot_pos = self.robot.get_position()
            dists = np.linalg.norm(obs_arr[:, :2] - robot_pos, axis=1)
            idx = int(np.argmin(dists))
            nearest_obs = obs_arr[idx]
            if dists[idx] <= self.trigger_distance:
                # front check
                heading = np.array([np.cos(self.robot.get_orientation()), np.sin(self.robot.get_orientation())])
                rel = nearest_obs[:2] - robot_pos
                if np.dot(rel, heading) > 0:
                    use_backup = True

        if use_backup:
            u_ref_val = self.compute_backup_ref()
        else:
            # control_ref may be a dict; if user passed u_ref param directly handle both
            if isinstance(control_ref, dict) and 'u_ref' in control_ref:
                u_ref_val = control_ref['u_ref']
            else:
                u_ref_val = control_ref

        # compute backup trajectory phi
        phi = self.compute_backup_trajectory(robot_state)

        # cache the controller time step
        dt = self.robot.dt

        # current dynamics evaluation
        A_list = []
        b_list = []

        # helper to safely compute barrier constraints for a state x and an obstacle
        def make_constraint(x_state, obs):
            if obs is None:
                return None
            # robot.agent_barrier(X, obs, robot_radius) returns (h, h_dot, dh_dot_dx)
            x_state = np.array(x_state, dtype=float).reshape(-1, 1)
            h, h_dot, dh_dot_dx = self.robot.robot.agent_barrier(x_state, obs, self.robot.robot_radius)
            g_mat = self.robot.robot.g(x_state)
            f_vec = self.robot.robot.f(x_state)
            Arow = (dh_dot_dx @ g_mat).reshape(-1)
            brow = (-(dh_dot_dx @ f_vec)
                    - (self.cbf_param['alpha1'] + self.cbf_param['alpha2']) * h_dot
                    - self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * h).reshape(-1, 1)
            return Arow, brow

        # current-state constraint (use nearest_obs if exists)
        if nearest_obs is not None:
            c = make_constraint(robot_state, nearest_obs)
            if c is not None:
                A_list.append(c[0])
                b_list.append(c[1])
        else:
            A_list.append(np.zeros((2,)))
            b_list.append(np.array([[1e6]]))

        # future constraints along phi
        for t in range(self.T):
            x_t = phi[t, :].reshape(-1, 1)
            if nearest_obs is not None:
                obs_future = nearest_obs.astype(float).copy()
                obs_future[0] += obs_future[3] * dt * (t + 1)
                obs_future[1] += obs_future[4] * dt * (t + 1)
                c = make_constraint(x_t, obs_future)
                if c is not None:
                    A_list.append(c[0])
                    b_list.append(c[1])
            else:
                A_list.append(np.zeros((2,)))
                b_list.append(np.array([[1e6]]))

        # fill parameter values
        A_mat = np.vstack(A_list)
        b_mat = np.vstack(b_list)
        # ensure shapes match parameter initializations
        self.A.value = A_mat
        self.b.value = b_mat
        self.u_ref.value = u_ref_val.reshape(2,1)

        # solve QP
        try:
            # prefer GUROBI if installed
            self.cbf_controller.solve(solver=cp.GUROBI, warm_start=True)
        except Exception:
            try:
                self.cbf_controller.solve(solver=cp.OSQP, warm_start=True)
            except Exception as e:
                self.status = 'failed'
                return u_ref_val.reshape(2,1)

        self.status = self.cbf_controller.status
        u_out = self.u.value
        if u_out is None:
            return u_ref_val.reshape(2,1)
        return u_out

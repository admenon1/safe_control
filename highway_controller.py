import numpy as np
import matplotlib.patches as patches

from tracking import LocalTrackingController
from position_control.backup_cbf_qp import BackupASIF


class HighwayController(LocalTrackingController):
    def __init__(self, X0, robot_spec, highway_env, **kwargs):
        kwargs.setdefault('env', highway_env)
        super().__init__(X0, robot_spec, **kwargs)
        self.highway_env = highway_env

        self._lane_patches = []
        self._traffic = []
        self._traffic_patches = []
        self._backup_traj_lines = [] 

        self.asif_filter = BackupASIF(self.robot, self.robot_spec)

    def init_traffic(self, traffic_spec):
        if not traffic_spec:
            self._traffic = []
            self._traffic_patches = []
            self.obs = np.empty((0, 7))
            return

        rows = []
        patches_list = []
        for car in traffic_spec:
            rows.append([car["x"], car["y"], car["radius"],
                         car.get("vx", 0.0), car.get("vy", 0.0), 0.0, 0.0])
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
        if not self._traffic:
            return

        for idx, car in enumerate(self._traffic):
            car["x"] += car.get("vx", 0.0) * self.dt
            car["y"] += car.get("vy", 0.0) * self.dt
            self.obs[idx, 0] = car["x"]
            self.obs[idx, 1] = car["y"]

            if idx < len(self._traffic_patches):
                circle = self._traffic_patches[idx]
                circle.center = (car["x"], car["y"])

    def control_step(self):
        self._advance_traffic()

        # constant-speed nominal input
        u_des = self.robot.nominal_input_constant_speed(
            target_speed=self.robot_spec.get('v_nominal', 2.0)
        ).flatten()

        # u_bound = self.asif_filter.u_max (Attempt to not let backup activate in the beginning)
        # u_des = np.clip(u_des, -u_bound, u_bound)

        # guard with the nearest obstacle
        if self.obs.size > 0:
            guard = self.obs[0]
            ego_pos = self.robot.get_position()
            dists = np.linalg.norm(self.obs[:, :2] - ego_pos, axis=1)
            guard = self.obs[np.argmin(dists)]
        else:
            guard = np.array([1e6, 0.0, 0.0, 0.0, 0.0])

        u_safe, intervening = self.asif_filter.asif(
            self.robot.X.flatten(), u_des, guard
        )

        self.robot.step(u_safe.reshape(-1, 1))

        if self.show_animation:
            self.robot.render_plot()

        return 0 if not intervening else 1

    def draw_plot(self, pause=0.01, force_save=False):
        if self.show_animation:
            for patch in self._lane_patches:
                try:
                    patch.remove()
                except Exception:
                    pass
            self._lane_patches.clear()

            # Clear previous backup trajectories
            for line in self._backup_traj_lines:
                try:
                    line.remove()
                except Exception:
                    pass
            self._backup_traj_lines.clear()

            # Draw the stores backup trajectories
            if hasattr(self, 'asif_filter') and self.asif_filter.visualize_backup:
                trajs = self.asif_filter.get_backup_trajectories()
                for phi in trajs:
                    line, = self.ax.plot(
                        phi[:, 0], phi[:, 1],
                        color='orange', linestyle='--', linewidth=1.0, alpha=0.7, zorder=2
                    )
                    self._backup_traj_lines.append(line)

            ego_x = float(self.robot.get_position()[0])
            lane_patches = self.highway_env.render_lanes(self.ax, ego_x=ego_x)
            if lane_patches:
                self._lane_patches.extend(lane_patches)

        super().draw_plot(pause=pause, force_save=force_save)
import numpy as np
import matplotlib.patches as patches

from tracking import LocalTrackingController
from utils.highway_env import HighwayEnv

class HighwayController(LocalTrackingController):
    def __init__(self, X0, robot_spec, highway_env: HighwayEnv, **kwargs):
        kwargs.setdefault('env', highway_env)
        super().__init__(X0, robot_spec, **kwargs)
        self.highway_env = highway_env
        # lane markings and moving traffic artists
        self._lane_patches = []
        self._traffic = []
        self._traffic_patches = []

    def _clear_lane_patches(self):
        for p in self._lane_patches:
            try:
                p.remove()
            except Exception:
                pass
        self._lane_patches = []

    def init_traffic(self, traffic_spec):
        """
        traffic_spec: list of dicts with keys
        {x, y, radius, vx} (vy is zero for highway lanes)
        """
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
                patches.Rectangle(
                    (car["x"] - 0.9, car["y"] - 0.4),
                    1.8, 0.8,
                    facecolor="gray",
                    edgecolor="black",
                    alpha=0.8,
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
                patch = self._traffic_patches[idx]
                patch.set_x(car["x"] - 0.9)
                patch.set_y(car["y"] - 0.4)

    def control_step(self):
        self._advance_traffic()
        return super().control_step()

    def draw_plot(self, pause=0.01, force_save=False):
        if self.show_animation:
            # remove only previous lane patches
            self._clear_lane_patches()
            ego_x = float(self.robot.get_position()[0])
            # render_lanes will now return the list of created patches (modify render_lanes accordingly)
            lane_patches = self.highway_env.render_lanes(self.ax, ego_x=ego_x)
            # store so we can remove them next frame
            if lane_patches is not None:
                self._lane_patches.extend(lane_patches)
            # call base draw to update robot, obstacles etc.
            super().draw_plot(pause=pause, force_save=force_save)
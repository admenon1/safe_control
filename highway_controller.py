from utils.highway_env import HighwayEnv
from tracking import LocalTrackingController
import numpy as np

class HighwayController(LocalTrackingController):
    def __init__(self, X0, robot_spec, highway_env: HighwayEnv, **kwargs):
        kwargs.setdefault('env', highway_env)
        super().__init__(X0, robot_spec, **kwargs)
        self.highway_env = highway_env
        # keep track of lane patches we add so we can remove only those
        self._lane_patches = []

    def _clear_lane_patches(self):
        for p in self._lane_patches:
            try:
                p.remove()
            except Exception:
                pass
        self._lane_patches = []

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
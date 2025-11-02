from utils.env import Env
import numpy as np
import matplotlib.patches as patches

class HighwayEnv(Env):
    def __init__(self, width=60.0, height=12.0, num_lanes=3, known_obs=[]):
        super().__init__(width=width, height=height, known_obs=known_obs)
        
        # Highway-specific parameters
        self.num_lanes = num_lanes
        self.lane_width = height / num_lanes
        self.lane_centers = [
            (i + 0.5) * self.lane_width 
            for i in range(num_lanes)
        ]
        # default safe area if not already set on parent/env
        if not hasattr(self, 'safe_area'):
            self.safe_area = {
                'x': width - 10.0,
                'y': self.lane_centers[-1],
                'w': 8.0,
                'h': self.lane_width * 0.8
            }

    def render_lanes(self, ax, ego_x=None):
        """
        Draw lane lines and safe area on provided axes.
        Returns list of created artists so caller can remove them next frame.
        """
        created = []

        # Update view limits to follow ego vehicle
        if ego_x is not None:
            margin = 15.0  # View margin
            ax.set_xlim(ego_x - margin, ego_x + margin)

        x0, x1 = ax.get_xlim()

        # Draw lane boundaries (solid + dashed for visibility)
        for i in range(self.num_lanes + 1):
            y = i * self.lane_width
            line1, = ax.plot([x0, x1], [y, y], color='black', linewidth=1.0, zorder=0)
            line2, = ax.plot([x0, x1], [y, y], color='blue', linestyle='--', linewidth=0.6, zorder=0)
            created.append(line1)
            created.append(line2)

        # Draw safe area rectangle if inside view
        sx = self.safe_area['x']
        if sx >= x0 - 1.0 and sx <= x1 + 1.0:
            safe_rect = patches.Rectangle(
                (self.safe_area['x'], self.safe_area['y'] - self.safe_area['h'] / 2),
                self.safe_area['w'],
                self.safe_area['h'],
                facecolor='lightgreen',
                edgecolor='green',
                alpha=0.35,
                zorder=1
            )
            ax.add_patch(safe_rect)
            created.append(safe_rect)

        return created
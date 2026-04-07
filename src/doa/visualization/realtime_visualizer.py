import numpy as np
import matplotlib.pyplot as plt
import os
import time

from src.core.utils.log_manager import create_log_directory

class RealtimeVisualizer:
    def __init__(self, config, sample_rate, num_channels, log_directory: str = None):
        self.config = config
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.frame_count = 0

        # Use provided log directory or create a new one
        if log_directory:
            self.log_dir = log_directory
        else:
            self.log_dir = create_log_directory()

        self.output_dir = os.path.join(self.log_dir, "realtime_plots")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Realtime plots will be saved to: {self.output_dir}")

        # Initialize plot elements once
        self.fig = None
        self.ax = None
        self.heatmap_line = None
        self.heatmap_fill = None
        self.doa_arrows = [] # List to hold multiple arrow objects
        
        if self.config.realtime_plot_enabled:
            self._setup_plot()
        
        print("RealtimeVisualizer Initialized.")

    def _setup_plot(self):
        """Sets up the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, polar=True)
        
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_rlabel_position(0)
        self.ax.set_thetagrids(np.arange(0, 360, 15))
        self.ax.set_rlim(0, 1.2)
        self.ax.set_title("DOA Spatial Spectrum", va='bottom')

        self.heatmap_line, = self.ax.plot([], [], color='b', alpha=0.7)

    def update_plot_and_save(self, data: dict):
        """
        Callback to update the combined plot and save it to a file.
        `data` is a dictionary containing spectrum_data and results list.
        """
        if not self.config.realtime_plot_enabled:
            return

        spectrum_data = data["spectrum_data"]
        results = data["results"] # List of (angle, energy) tuples

        angles_deg, spectrum = spectrum_data
        angles_rad = np.deg2rad(angles_deg)

        # Update heatmap
        self.heatmap_line.set_data(angles_rad, spectrum)
        if self.heatmap_fill:
            self.heatmap_fill.remove()
        self.heatmap_fill = self.ax.fill_between(angles_rad, 0, spectrum, color='b', alpha=0.3)

        # Update DOA arrows
        for arrow in self.doa_arrows:
            arrow.remove()
        self.doa_arrows.clear()

        title_angles = []
        for angle, energy in results:
            angle_rad = np.deg2rad(angle)
            arrow = self.ax.annotate(
                '', xy=(angle_rad, 1.1), xytext=(angle_rad, 0.5),
                arrowprops=dict(facecolor='red', shrink=0.1, width=2, headwidth=10)
            )
            self.doa_arrows.append(arrow)
            title_angles.append(f"{angle:.1f}°")

        self.ax.set_title(f"DOA Frame {self.frame_count}, Angles: {', '.join(title_angles) or 'None'}")

        filename = os.path.join(self.output_dir, f"frame_{self.frame_count:05d}.png")
        self.fig.savefig(filename)
        self.frame_count += 1

    def close(self):
        """Closes the matplotlib figure."""
        if self.fig:
            plt.close(self.fig)
            print("RealtimeVisualizer figure closed.")

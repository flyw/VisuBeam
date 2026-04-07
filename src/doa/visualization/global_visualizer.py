import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time

from src.core.utils.log_manager import create_log_directory

class GlobalVisualizer:
    def __init__(self, config, sample_rate, num_channels, log_directory: str = None):
        self.config = config
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        self.srp_spectrums = []
        self.results = []
        self.start_time = None

        # Use provided log directory or create a new one
        if log_directory:
            self.log_dir = log_directory
        else:
            self.log_dir = create_log_directory()

        print("GlobalVisualizer Initialized.")

    def accumulate_data(self, data: dict):
        """Callback to accumulate all necessary data for global plots."""

        if self.start_time is None:
            self.start_time = data["timestamp"]

        # Calculate relative time for this chunk
        relative_time = data["timestamp"] - self.start_time

        if self.config.global_heatmap_enabled:
            # The spectrum data is already in dB.
            _, spectrum = data["spectrum_data"]
            self.srp_spectrums.append(spectrum)

        if self.config.global_doa_plot_enabled:
            # data["results"] is a list of (angle, energy) tuples
            for angle, energy in data["results"]:
                self.results.append((relative_time, angle, energy))

    def close(self):
        """
        Processes accumulated data and saves global plots based on user's reference code.
        """
        print("GlobalVisualizer closing. Generating global plots...")

        if not self.results and not self.srp_spectrums:
            print("No data accumulated for global plots.")
            return

        # Estimate total duration from the last timestamp
        total_duration = self.results[-1][0] if self.results else 0
        processed_duration = total_duration if self.srp_spectrums else 0

        # 1. Plot SRP Heatmap (as per user's code)
        if self.config.global_heatmap_enabled and self.srp_spectrums:
            srp_heatmap_db = np.array(self.srp_spectrums).T
            
            plt.figure(figsize=(12, 7))
            ax_heatmap = plt.gca()

            if srp_heatmap_db.size > 0:
                vmax = np.max(srp_heatmap_db)
                vmin = vmax - 20  # 20dB dynamic range
            else:
                vmax, vmin = 0, -60

            im = plt.imshow(srp_heatmap_db, aspect='auto', origin='lower',
                            extent=[0, processed_duration, 0, 180],
                            interpolation='bilinear', vmin=vmin, vmax=vmax)
            plt.colorbar(im, label='SRP Power (dB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (°)')
            plt.title('SRP-PHAT Heatmap Over Time')
            ax_heatmap.set_xlim(0, total_duration)

            if not os.path.exists(self.log_dir):
                print(f"GlobalVisualizer: Log directory {self.log_dir} missing, recreating.")
                os.makedirs(self.log_dir, exist_ok=True)

            filepath = os.path.join(self.log_dir, 'srp_heatmap.png')
            plt.savefig(filepath)
            print(f"Global SRP heatmap saved to {filepath}")
            plt.close()

        # 2. Plot DOA Scatter (as per user's code)
        if self.config.global_doa_plot_enabled and self.results:
            # All results reaching here are assumed to be valid after upstream filtering
            filtered_results = self.results

            if not filtered_results:
                print(f"No DOA results found to plot.")
                print("GlobalVisualizer closed.")
                return

            print(f"Plotting {len(filtered_results)} DOA results.")
            times = [r[0] for r in filtered_results]
            angles = [r[1] for r in filtered_results]
            energies = [r[2] for r in filtered_results]

            plt.figure(figsize=(12, 7))
            ax_scatter = plt.gca()
            plt.scatter(times, angles, c=energies, cmap='viridis', s=1, alpha=0.7)
            plt.colorbar(label='Peak Energy')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (°)')
            plt.title('DOA Results Over Time')
            plt.ylim(0, 180)
            ax_scatter.set_xlim(0, total_duration)
            plt.grid(True)

            if not os.path.exists(self.log_dir):
                print(f"GlobalVisualizer: Log directory {self.log_dir} missing, recreating.")
                os.makedirs(self.log_dir, exist_ok=True)

            filepath = os.path.join(self.log_dir, 'doa_plot.png')
            plt.savefig(filepath)
            print(f"Global DOA plot saved to {filepath}")
            plt.close()

        print("GlobalVisualizer closed.")

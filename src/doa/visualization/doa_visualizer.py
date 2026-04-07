import numpy as np
import matplotlib.pyplot as plt
import queue
import time

class DOAVisualizer:
    def __init__(self, config=None):
        """
        Initializes the DOA Visualizer. This version uses a queue to allow
        matplotlib to run safely in the main thread.
        """
        self.config = config
        self.plot_queue = queue.Queue()
        self.is_running = True
        
        # Plot elements are initialized here but only manipulated in the main loop
        self.fig = None
        self.ax = None
        self.heatmap_line = None
        self.heatmap_fill = None
        self.doa_arrow = None
        
        print("DOA Visualizer Initialized.")

    def _initialize_plot(self):
        """Initializes the matplotlib figure and axes. Must be called from the main thread."""
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, polar=True)
        
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_rlabel_position(0)
        self.ax.set_thetagrids(np.arange(0, 360, 15))
        self.ax.set_rlim(0, 1.2)
        self.ax.set_title("DOA Spatial Spectrum", va='bottom')

        self.heatmap_line, = self.ax.plot([], [], color='b', alpha=0.7)
        plt.show(block=False)

    def update_heatmap(self, spectrum_data: tuple):
        """Callback to queue heatmap data."""
        self.plot_queue.put(('heatmap', spectrum_data))

    def update_doa_plot(self, angle: float):
        """Callback to queue DOA angle data."""
        self.plot_queue.put(('doa', angle))

    def start_visualization_loop(self):
        """
        This method should be run in the main thread. It polls the queue
        for data and updates the plot.
        """
        self._initialize_plot()
        
        while self.is_running:
            try:
                # Process all pending items in the queue
                while not self.plot_queue.empty():
                    item_type, data = self.plot_queue.get_nowait()
                    
                    if item_type is None: # Sentinel for closing
                        self.is_running = False
                        break

                    if item_type == 'heatmap':
                        self._draw_heatmap(data)
                    elif item_type == 'doa':
                        self._draw_doa_arrow(data)
                
                if not self.is_running:
                    break

                # Redraw the canvas
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                time.sleep(0.1) # Small sleep to yield CPU

            except KeyboardInterrupt:
                self.is_running = False
        
        self._close_plot()

    def _draw_heatmap(self, spectrum_data):
        angles_deg, spectrum = spectrum_data
        angles_rad = np.deg2rad(angles_deg)
        self.heatmap_line.set_data(angles_rad, spectrum)
        if self.heatmap_fill:
            self.heatmap_fill.remove()
        self.heatmap_fill = self.ax.fill_between(angles_rad, 0, spectrum, color='b', alpha=0.3)

    def _draw_doa_arrow(self, angle):
        if self.doa_arrow:
            self.doa_arrow.remove()
        angle_rad = np.deg2rad(angle)
        self.doa_arrow = self.ax.annotate(
            '', xy=(angle_rad, 1.1), xytext=(angle_rad, 0.5),
            arrowprops=dict(facecolor='red', shrink=0.1, width=2, headwidth=10)
        )

    def _close_plot(self):
        """Closes the matplotlib window."""
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            print("DOA Visualizer plot closed.")

    def close(self):
        """Signals the visualization loop to stop."""
        print("Closing DOA Visualizer...")
        if self.is_running:
            self.plot_queue.put((None, None)) # Sentinel

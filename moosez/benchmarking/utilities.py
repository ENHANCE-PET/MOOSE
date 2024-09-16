import time
import psutil
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy


colors = [
    (1, 1, 1),  # white
    (0.894, 0.102, 0.11),  # red
    (0.216, 0.494, 0.722),  # blue
    (0.302, 0.686, 0.29),  # green
    (0.596, 0.306, 0.639),  # purple
    (1.0, 0.498, 0.0),  # orange
    (1.0, 1.0, 0.2),  # yellow
    (0.651, 0.337, 0.157),  # brown
    (0.6, 0.6, 0.6),  # gray
    (0.5, 0.5, 0),  # olive
    (0, 0.5, 0.5),  # teal
    (0.5, 0, 0.5),  # purple
    (0.5, 0.5, 1),  # light blue
    (0.9, 0.1, 0.7),  # magenta
    (0.9, 0.7, 0.1),  # gold
    (0.1, 0.9, 0.9),  # cyan
    (0.9, 0.4, 0.3),  # coral
    (0.3, 0.9, 0.4),  # light green
    (0.7, 0.7, 0.7),  # light gray
    (0.8, 0.6, 0.2),  # mustard
    (0.4, 0.2, 0.6),  # plum
    (0.2, 0.6, 0.8),  # sky blue
    (0.6, 0.2, 0.4),  # raspberry
    (0.7, 0.5, 0.9),  # lavender
    (0.4, 0.7, 0.1),  # lime
    (0.9, 0.2, 0.4),  # rose
    (0.1, 0.7, 0.6),  # turquoise
    (0.9, 0.8, 0.6),  # beige
    (0.7, 0.1, 0.3),  # maroon
    (0.5, 0.7, 0.9),  # light sky blue
    (0.8, 0.4, 0.7),  # orchid
    (0.3, 0.9, 0.5),  # mint
    (0.6, 0.1, 0.9),  # violet
    (0.9, 0.9, 0.2),  # lemon
    (0.4, 0.6, 0.9),  # periwinkle
    (0.7, 0.9, 0.1),  # chartreuse
    (0.9, 0.4, 0.2),  # tangerine
    (0.2, 0.9, 0.7),  # aqua
    (0.9, 0.2, 0.9),  # fuchsia
]

# Create the custom colormap
n_bins = 256  # Number of bins in the colormap
cmap_name = 'white_to_colors'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


class PerformanceObserver:
    def __init__(self, image: str | None = None, model: str | None = None):
        self.process_start_time = time.time()
        self.total_runtime = None
        self.running =  True

        self.memory_timestamps = []
        self.memory_usage_MB = []
        self.memory_usage_GB =  []

        self.phase_timestamps = []
        self.phase_names = []
        self.phase_runtimes = []

        self.metadata_image = image
        self.metadata_image_size = None
        self.metadata_model = model

    def __get_memory_usage_of_process_tree(self):
        parent = psutil.Process(os.getpid())
        memory_usage = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                memory_usage += child.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return memory_usage / (1024 * 1024)  # Convert to MB

    def monitor_memory_usage(self, interval):
        while self.running:
            current_time = time.time() - self.process_start_time
            current_memory_MB = self.__get_memory_usage_of_process_tree()
            current_memory_GB = current_memory_MB / 1024
            self.memory_timestamps.append(current_time)
            self.memory_usage_MB.append(current_memory_MB)
            self.memory_usage_GB.append(current_memory_GB)
            time.sleep(interval)

    def record_phase(self, phase_name: str):
        current_time = time.time() - self.process_start_time
        self.phase_timestamps.append(current_time)
        self.phase_names.append(phase_name)

    def time_phase(self):
        if not self.phase_names:
            start_time = self.process_start_time
        else:
            start_time = self.phase_timestamps[-1]
        runtime = (time.time() -  self.process_start_time) - start_time
        self.phase_runtimes.append(runtime)

    def plot_performance(self, path: str):
        fig, axs = plt.subplots(2, 1, figsize=(22, 12), gridspec_kw={'height_ratios': [3, 1]})

        # Plot the memory usage over time
        axs[0].plot(self.memory_timestamps, self.memory_usage_GB, color='lightblue', label='Memory Usage (GB)')
        axs[0].set_xlabel('Time [s]', fontsize=16)
        axs[0].set_xticks(numpy.arange(0, max(self.memory_timestamps) + 1, 5))
        axs[0].set_ylim(0, 32)
        axs[0].set_ylabel('Memory Usage [GB]', fontsize=16)
        axs[0].set_yticks([0, 8, 16, 24, 32])
        axs[0].tick_params(axis='y', labelsize=16)
        axs[0].tick_params(axis='x', labelsize=16)
        image_name = os.path.basename(self.metadata_image)
        axs[0].set_title(f'Memory Usage Over Time.\nImage: {image_name} | Model: {self.metadata_model}', fontsize=20)
        axs[0].grid(axis='x')

        for i, (timestamp, phase_name) in enumerate(zip(self.phase_timestamps, self.phase_names), start=1):
            axs[0].axvline(x=timestamp, color='lightpink', linestyle='-', label=f'Phase {i}')
            axs[0].text(timestamp + 0.3, axs[0].get_ylim()[1] - 1.3, str(i), rotation=0, verticalalignment='bottom',
                        fontsize=16, ha='center')

        # Plot the phase times as a horizontal bar graph
        phase_labels = [f'{i + 1}' for i in range(len(self.phase_names))]
        bars = axs[1].barh(phase_labels[:-1], self.phase_runtimes, color='lightblue')
        axs[1].set_xticks(numpy.arange(0, max(self.phase_runtimes) + 1, 5))
        axs[1].set_xlabel('Time [s]', fontsize=16)
        axs[1].set_ylabel('Phases', fontsize=16)
        axs[1].invert_yaxis()
        axs[1].tick_params(axis='x', labelsize=16)
        axs[1].tick_params(axis='y', labelsize=16)
        axs[1].grid(axis='x')

        # Add phase names inside the bars, aligned to the left with black text
        for bar, phase_name in zip(bars, self.phase_names):
            axs[1].text(bar.get_x() + 0.1, bar.get_y() + bar.get_height() / 2, phase_name, ha='left', va='center',
                        fontsize=16, color='black')

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'performance_plot.png'))

    def off(self):
        self.running = False
        self.total_runtime = time.time() - self.process_start_time

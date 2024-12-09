import time
import psutil
import matplotlib.pyplot as plt
import os
import numpy
import threading
from typing import Union, Tuple, List, Dict


class PerformanceObserver:
    def __init__(self, image: Union[str, None] = None, model: Union[str, None] = None, polling_rate: float = 0.1):
        self.monitoring = False
        self.polling_rate = polling_rate
        self.monitoring_thread = None
        self.monitoring_start_time = None
        self.total_runtime = None

        self.memory_timestamps = []
        self.memory_usage_MB = []
        self.memory_usage_GB =  []

        self.phase_timestamps = []
        self.phase_names = []
        self.phase_runtimes = []

        self.metadata_image = image
        self.metadata_image_size = None
        self.metadata_model = model

    def on(self):
        self.monitoring = True
        self.monitoring_start_time = time.time()
        self.monitoring_thread = threading.Thread(target=self.__monitor_memory_usage, args=(self.polling_rate,))
        self.monitoring_thread.start()

    def off(self):
        self.monitoring = False
        if self.monitoring_thread is not None:
            self.monitoring_thread.join()
        self.total_runtime = time.time() - self.monitoring_start_time

    def __get_memory_usage_of_process_tree(self):
        parent = psutil.Process(os.getpid())
        memory_usage = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                memory_usage += child.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return memory_usage / (1024 * 1024)  # Convert to MB

    def __monitor_memory_usage(self, interval: float):
        while self.monitoring:
            current_time = time.time() - self.monitoring_start_time
            current_memory_MB = self.__get_memory_usage_of_process_tree()
            current_memory_GB = current_memory_MB / 1024
            self.memory_timestamps.append(current_time)
            self.memory_usage_MB.append(current_memory_MB)
            self.memory_usage_GB.append(current_memory_GB)
            time.sleep(interval)

    def record_phase(self, phase_name: str):
        if self.monitoring:
            current_time = time.time() - self.monitoring_start_time
            self.phase_timestamps.append(current_time)
            self.phase_names.append(phase_name)

    def time_phase(self):
        if self.monitoring:
            if not self.phase_names:
                start_time = self.monitoring_start_time
            else:
                start_time = self.phase_timestamps[-1]
            runtime = (time.time() - self.monitoring_start_time) - start_time
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
        plt.close()

    def get_peak_resources(self) -> List:
        image_name = os.path.basename(self.metadata_image)
        model_name = self.metadata_model
        if self.metadata_image_size is not None and isinstance(self.metadata_image_size, (list, tuple)):
            image_size = "x".join(map(str, self.metadata_image_size))
        else:
            image_size = "unknown"
        if self.total_runtime:
            runtime = self.total_runtime
        else:
            runtime = 0
        if self.memory_usage_GB:
            memory_peak = max(self.memory_usage_GB)
        else:
            memory_peak = 0

        return [image_name, model_name, image_size, runtime, memory_peak]
import os
import threading
import time
import logging
import cProfile
import pstats
import io
import psutil
import torch


class SingletonMeta(type):
    """Metaclass for singleton pattern"""
    _instances = {}

    def __call__(cls):
        return cls._instances[cls]

    def create_singleton_instance(cls, *args, **kwargs):
        if cls in cls._instances:
            raise ValueError(f"{cls} instance already exists")
        cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)

    def clear_singleton_instance(cls):
        del cls._instances[cls]


class Profiler(threading.Thread, metaclass=SingletonMeta):
    """
    Threaded class to profile any python process using a singleton instance.
    Must be initialized with:
        Profile.create_singleton_instance(*args)
        my_prof = Profile()
    And stopped/destroyed using:
        my_prof.stop()
        Profile.clear_singleton_instance()
    """
    def __init__(self, filename=None, delay_in_ms=100, log_level=logging.NOTSET):
        self.__stop = False
        self.filename = filename
        self.delay_in_ms = delay_in_ms
        self.log_level = log_level

        self.suffix = ".tsv"
        self.msg = None
        self._section = " "
        self._loop_step = " "
        self.gpu_avail = None
        self.use_file_handler = None
        self.filepath = None
        self.cprof = None

        if self.log_level > logging.NOTSET:
            logging.info(" ")
            logging.info("Starting profiler")
            logging.info(" ")

            self.gpu_avail = torch.cuda.is_available()
            self.use_file_handler = (self.filename is not None)
            if self.use_file_handler:
                self.filepath = os.path.join(os.getcwd(), self.filename + self.suffix)
            self.cprof = cProfile.Profile()
            # cProfile must be started now
            # to avoid profiling only current profiler thread
            self.cprof.enable()
            self._create_usage_desc()

            threading.Thread.__init__(self)
            self.start()

    def _create_usage_desc(self):
        """
        Create the usage descriptor (column names) in tsv format
        and optionnaly write it to a file
        """
        usage_desc = ("time (s)\tsection (str)\tloop_step (str)"
                      "\tcpu_usage (%)\tmem_used (GB)\tmem_total (GB)"
                      "\tmem_read_bytes (GB)\tmem_write_bytes (GB)")
        if self.gpu_avail:
            usage_desc += "\tgpu_name (str)\tgpu_load (%)\tgpu_mem_used (GB)\tgpu_mem_total (GB)"
        self.usage_desc = usage_desc

        if self.use_file_handler:
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(self.usage_desc)
        if self.log_level > logging.DEBUG:
            print(f"{self.usage_desc}")

    def _log_profile(self):
        """Main logger for Profiler class"""
        if self.use_file_handler:
            with open(self.filepath, mode="a") as f:
                f.write(self.msg)
        if self.log_level > logging.DEBUG:
            print(f"{self.msg}")

    def _log_cprofile(self):
        """Logger from cProfile module"""
        if self.use_file_handler:
            result = io.StringIO()
            stats = pstats.Stats(self.cprof, stream=result)
            stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats()
            stats_tsv = "ncalls" + result.getvalue().split('ncalls')[-1]
            stats_tsv = "\n".join(["\t".join(line.rstrip().split(None, 5)) for line in stats_tsv.split("\n")])
            with open(f"{self.filepath[:-4]}_cprofile" + self.suffix, "w+") as f:
                f.write(stats_tsv)
        if self.log_level > logging.DEBUG:
            self.cprof.print_stats()

    def get_msg(self):
        """Return the hardware usage message from the profiler"""
        return self.msg

    def set_section(self, section):
        """Defines a generic label for a section in the algorithm"""
        self._section = section

    def set_loop_step(self, loop_step):
        """Defines a label for a loop step"""
        self._loop_step = loop_step

    def stop(self):
        """Used to stop the thread"""
        self.__stop = True

    def run(self):
        """Main thread loop"""
        start_time = time.time()
        gpu_full_name = torch.cuda.get_device_name()
        while True:
            if self.__stop:
                break
            # time
            curr_time = time.time()
            elapsed = curr_time - start_time
            self.msg = f"\n{elapsed:.6f}"
            # section and loop step
            self.msg += f"\t{self._section}\t{self._loop_step}"
            # CPU and RAM usage
            mem_usage = psutil.virtual_memory()
            self.msg += f"\t{psutil.cpu_percent()}\t{get_size(mem_usage.used)}\t{get_size(mem_usage.total)}"
            # Disk IO
            disk_io = psutil.disk_io_counters()
            self.msg += f"\t{get_size(disk_io.read_bytes)}\t{get_size(disk_io.write_bytes)}"
            # GPU
            if self.gpu_avail:
                gpu_mem_free, gpu_mem_total = torch.cuda.mem_get_info()
                gpu_load = torch.cuda.utilization()
                gpu_mem_used = gpu_mem_total - gpu_mem_free
                gpu_mem_used_gb = get_size(gpu_mem_used)
                gpu_mem_total_gb = get_size(gpu_mem_total)
                self.msg += f"\t{gpu_full_name}\t{gpu_load}\t{gpu_mem_used_gb}\t{gpu_mem_total_gb}"
            self._log_profile()
            time.sleep(self.delay_in_ms/1000)
        self.cprof.disable()
        self._log_cprofile()


def get_size(bytes, unit="G"):
    """Scale bytes given the unit"""
    units = " KMGTP"
    power = units.find(unit)
    factor = 1024**power
    num = round(bytes / factor, 3)

    return num


if __name__ == '__main__':
    Profiler.create_singleton_instance(filename="test", log_level=logging.DEBUG)
    t1 = Profiler()
    time.sleep(1)
    t1.stop()
    Profiler.clear_singleton_instance()

    Profiler.create_singleton_instance(log_level=logging.INFO)
    t2 = Profiler()
    time.sleep(1)
    t2.stop()
    Profiler.clear_singleton_instance()

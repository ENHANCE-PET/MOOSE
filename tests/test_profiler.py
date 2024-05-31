import pytest
import time
import logging
from moosez.benchmarking.profiler import Profiler


@pytest.fixture(autouse=True)
def profiler_setup():
    Profiler.create_singleton_instance(log_level=logging.DEBUG)
    prof = Profiler()
    yield
    prof.stop()
    Profiler.clear_singleton_instance()


@pytest.mark.unit
def test_profiler_start():
    thread = Profiler()
    assert thread.is_alive()


@pytest.mark.unit
def test_another_profiler_start():
    another_profiler = Profiler()

    with pytest.raises(RuntimeError) as excinfo:
        another_profiler.start()
    assert str(excinfo.value) == "threads can only be started once"


@pytest.mark.unit
def test_profiler_has_msg():
    thread = Profiler()
    time.sleep(0.25)

    assert thread.get_msg() is not None


@pytest.mark.unit
def test_profiler_is_singleton():
    one_profiler = Profiler()
    another_profiler = Profiler()

    assert one_profiler is another_profiler

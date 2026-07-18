import time


class LatencyTimer:
    """
    Measures API latency in milliseconds.
    """

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.start_time is None:
            return 0.0
        end = time.perf_counter()
        return round((end - self.start_time) * 1000, 2)

import time

class SolveIVPProgressWrapper:
    """
    A wrapper to show the progress of the solve_ivp function in %
    """
    def __init__(self, rhs_func, t_end, report_step_percent = 1):
        self.rhs_func = rhs_func
        self.t_end = float(t_end)
        self.report_step = report_step_percent
        self.last_reported_percent = 0
        self.start_time = time.time()

    def __call__(self, t, y):
        if self.t_end > 0:
            percent_done = (t / self.t_end) * 100
            if percent_done >= self.last_reported_percent + self.report_step:
                self.last_reported_percent = int(percent_done / self.report_step) * self.report_step
                elapsed_time = time.time() - self.start_time
                # estimate of expected time to end integration
                eta_seconds = (elapsed_time / percent_done * (100 - percent_done)) if percent_done > 0 else float('inf')
                print(f" Solver Progress: {self.last_reported_percent:.0f}% (t={t:.2f}, ETA: {eta_seconds:.1f}s)  ", end = '\r')
        return self.rhs_func(t, y)

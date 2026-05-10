def pick_entropy_band_task(task_to_success_rate, low=0.3, high=0.7):
    in_band = [k for k, v in task_to_success_rate.items() if low <= v <= high]
    if in_band:
        return in_band[0]
    return min(task_to_success_rate, key=lambda k: abs(task_to_success_rate[k] - 0.5))

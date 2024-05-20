import numpy as np


class NoiseSchedule:
    def __init__(self, steps: int):
        self.steps = steps
        self.schedule = self._generate_noise_schedule(steps)
        self.cumulative_product_noise = np.cumprod(1.0 - self.schedule, axis=0)
        self.sqrt_cumulative_product_noise = np.sqrt(self.cumulative_product_noise)
        self.sqrt_one_minus_cumulative_product_noise = np.sqrt(
            1.0 - self.cumulative_product_noise
        )

    def _generate_noise_schedule(self, timesteps: int):
        return linear_noise_schedule(timesteps)


def get_noisy_sample(dataset: np.ndarray, step: int, noise_schedule: NoiseSchedule):
    return noise_schedule.sqrt_cumulative_product_noise[
        step
    ] * dataset + noise_schedule.sqrt_one_minus_cumulative_product_noise[
        step
    ] * np.random.normal(
        size=dataset.shape
    )


def linear_noise_schedule(steps):
    beta_start = 0.0001
    beta_end = 0.02
    return np.linspace(beta_start, beta_end, steps)

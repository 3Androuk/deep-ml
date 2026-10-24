import numpy as np

class StepLRScheduler:
    def __init__(self, initial_lr, step_size, gamma):
        # Initialize initial_lr, step_size, and gamma
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch):
        # Calculate and return the learning rate for the given epoch
        lr = self.initial_lr * self.gamma ** np.floor(epoch / self.step_size)
        epsilon = 1e-12
        return round(lr + epsilon, 4)
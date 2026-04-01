import math
import numpy as np
import torch


def check(input_value):
    if isinstance(input_value, np.ndarray):
        return torch.from_numpy(input_value)
    return input_value


def get_gard_norm(parameters):
    total = 0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.norm() ** 2
    return math.sqrt(total)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def huber_loss(error, delta):
    small = (abs(error) <= delta).float()
    large = (abs(error) > delta).float()
    return small * error ** 2 / 2 + large * delta * (abs(error) - delta / 2)


def mse_loss(error):
    return error ** 2 / 2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        return obs_space.shape
    if obs_space.__class__.__name__ == 'list':
        return obs_space
    raise NotImplementedError


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        return 1
    if act_space.__class__.__name__ == 'MultiDiscrete':
        return act_space.shape
    if act_space.__class__.__name__ == 'Box':
        return act_space.shape[0]
    if act_space.__class__.__name__ == 'MultiBinary':
        return act_space.shape[0]
    return act_space[0].shape[0] + 1

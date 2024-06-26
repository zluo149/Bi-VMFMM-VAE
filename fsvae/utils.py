# encoding: utf-8
try:
    import math
    import torch
    import argparse
    import numpy as np
    import torch.nn.functional as F
    import torch.nn as nn

    from torchvision.utils import save_image
    from scipy.optimize import linear_sum_assignment as linear_assignment
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    from scipy.special import iv

    from fsvae.config import DEVICE, DATASETS_DIR

except ImportError as e:
    print(e)
    raise ImportError


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def init_params(data, a=0, mode='fan_in', nonlinearity='relu', type='1', std=0.02, gain=1):

    if type == 'k':
        fan = nn.init._calculate_correct_fan(data, mode)
        gain = nn.init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)

    if type == 'x':
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(data)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    # print("std: {}".format(std))
    with torch.no_grad():
        return data.normal_(0, std)


def init_weights(net):

    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # nn.init.xavier_normal_(m.weight)
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def save_images(gen_imgs, imags_path, images_name, nrow=5):

    save_image(gen_imgs.data[:nrow * nrow],
               '%s/%s.png' % (imags_path, images_name),
               nrow=nrow, normalize=True)


def cluster_acc(Y_pred, Y):

    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total*1.0/Y_pred.size, w

def mse(origin, target):

    loss = F.mse_loss(origin, target)

    return loss

# def besseli(nu, x):
#
#     frac = x / nu
#     square = 1 + frac**2
#     root = torch.sqrt(square)
#     eta = root + torch.log(frac) - torch.log(1 + root)
#     approx = -torch.log(torch.sqrt(torch.tensor(2 * np.pi * nu, dtype=torch.float).to(DEVICE))) + nu * eta - 0.25*torch.log(square)
#
#     return torch.exp(approx)

def log_besseli(nu, x):

    frac = x / nu
    square = 1 + frac**2
    root = torch.sqrt(square)
    eta = root + torch.log(frac) - torch.log(1 + root)
    log_approx = -torch.log(torch.sqrt(torch.tensor(2 * np.pi * nu, dtype=torch.float).to(DEVICE))) + nu * eta - 0.25*torch.log(square)

    return log_approx


# def d_besseli(nu, kk):
#
#     try:
#         bes = besseli(nu + 1, kk) / (besseli(nu, kk) + 1e-10)
#         assert (min(torch.isfinite(bes)))
#     except:
#         bes = torch.sqrt(1 + (nu**2) / (kk**2))
#
#     return bes

def d_besseli(v, z, eps=1e-20):
    # Ensure eps is on the same device as z
    eps_tensor = torch.tensor(eps, dtype=z.dtype, device=z.device)

    def delta_a(a):
        lamb = v + (a - 1.0) / 2.0
        return (v - 0.5) + lamb / (
            2 * torch.sqrt((torch.pow(lamb, 2) + torch.pow(z, 2)).clamp(min=eps_tensor))
        )

    delta_0 = delta_a(torch.tensor(0.0, dtype=z.dtype, device=z.device))
    delta_2 = delta_a(torch.tensor(2.0, dtype=z.dtype, device=z.device))
    B_0 = z / (
        delta_0 + torch.sqrt((torch.pow(delta_0, 2) + torch.pow(z, 2))).clamp(min=eps_tensor)
    )
    B_2 = z / (
        delta_2 + torch.sqrt((torch.pow(delta_2, 2) + torch.pow(z, 2))).clamp(min=eps_tensor)
    )

    return (B_0 + B_2) / 2.0


def sample_from_gamma(alpha):

    alpha = alpha + 1
    e = torch.tensor(np.random.normal(0, 1, alpha.size()), dtype=torch.float).to(DEVICE)
    return (alpha - 1 / 3) * torch.pow(1 + (e / torch.sqrt(9 * alpha - 3)), 3)
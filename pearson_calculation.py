# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    Pearson correlation coefficient calculation
# ----------------------------------
import torch
from audtorch.metrics import PearsonR


def PCC(Lap, Y_S, Wei, dist):
    """
    Calculate Pearson correlation coefficient (PCC) between two datasets.

    :param Lap: Laplacian matrix.
    :type Lap: torch.Tensor
    :param Y_S: Y_S description.
    :type Y_S: torch.Tensor
    :param Wei: Wei description.
    :type Wei: torch.Tensor
    :param dist: dist description.
    :type dist: torch.Tensor
    :return: Absolute value of Pearson correlation coefficient.
    :rtype: torch.Tensor
    """

    D_I = torch.matmul(Y_S, Wei[0, ::])
    D_L = torch.matmul(Lap, torch.matmul(Y_S, Wei[1, ::]))

    D1_abs, D2_abs = torch.abs(D_L), torch.abs(D_I)

    D1_mean, D2_mean = torch.mean(D1_abs, dim=2), torch.mean(D2_abs, dim=2)

    D_div = torch.div(D1_mean, D2_mean)

    D_p = D_div[:, 1:]
    D_pp = D_p.transpose(1, 0)

    D_pp_re, dist_re = torch.reshape(D_pp, (1, -1)), torch.reshape(dist, (1, -1))

    PR_cal = PearsonR()
    pear = PR_cal(D_pp_re, dist_re)

    return torch.abs(pear)

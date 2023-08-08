# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    DAGNN model definition
# ----------------------------------
import torch
import torch.nn as nn
import torch.nn.init as init
from pearson_calculation import PCC


class DAGNN_Conv(nn.Module):
    """
    Define the Graph Convolution Layer class (DAGNN_Conv).
    """

    def __init__(self, in_c, out_c, bias=True, normalize=True):
        super(DAGNN_Conv, self).__init__()
        self.normalize = normalize  # Normalization parameter, True or False

        self.weight = nn.Parameter(torch.Tensor(2, 1, in_c, out_c))
        init.xavier_normal_(self.weight)  # Initialize using Xavier normal distribution

        if bias:  # Bias, equivalent to the 'b' term in a linear function
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)  # Initialize with zeros
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs, graph, Y_data, Dist_data):
        L = DAGNN_Conv.get_laplacian(graph, self.normalize)  # Get the Laplacian matrix [N, N]

        mul_L = self.multi_polynomial(L).unsqueeze(1)  # Compute Multi-Polynomial T(L)

        result = torch.matmul(mul_L, inputs)  # Calculate [2, B, N, C]
        result = torch.matmul(result, self.weight)  # Calculate [2, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # Sum and add bias [B, N, D]

        if Y_data is not None:
            pearson = PCC(Lap=L, Y_S=Y_data, Wei=self.weight, dist=Dist_data)
            return result, pearson
        else:
            return result

    def multi_polynomial(self, laplacian):
        """
        Compute the Multi-Polynomial T(L).

        :param laplacian: the graph Laplacian, [N, N].
        :return: the multi-order Laplacian, [2, N, N].
        """
        N = laplacian.size(0)  # Number of nodes
        multi_order_laplacian = torch.zeros([2, N, N], device=laplacian.device, dtype=torch.float)
        multi_order_laplacian[0] = torch.eye(
            N, device=laplacian.device, dtype=torch.float
        )  # 0th order is the identity matrix
        multi_order_laplacian[1] = laplacian

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        Compute the graph Laplacian.

        :param graph: the graph structure without self-loop, [N, N].
        :param normalize: whether to use the normalized Laplacian.
        :return: graph Laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))  # Diagonal matrix for normalization
            return torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(
                torch.mm(D, graph), D
            )  # Normalized Laplacian
        else:
            D = torch.diag(torch.sum(graph, dim=-1))  # Diagonal matrix without normalization
            return D - graph  # Unnormalized Laplacian


class DAGNN_Net(nn.Module):
    """
    Define the Graph Neural Network class (DAGNN_Net).
    """

    def __init__(self, in_c, hid_c, out_c, num_node):
        super(DAGNN_Net, self).__init__()
        self.conv1 = DAGNN_Conv(in_c=in_c, out_c=hid_c)  # First graph convolution layer
        self.conv2 = DAGNN_Conv(in_c=hid_c, out_c=out_c)  # Second graph convolution layer
        self.linear3 = nn.Linear(num_node, out_c)  # Define a linear layer
        self.act = nn.ReLU()  # Activation function
        self.act_pear = nn.Sigmoid()  # Sigmoid activation function for Pearson Correlation Coefficient

    def forward(self, data, device, Y_data, Dist_data):
        graph_data = data["graph"].to(device)[0]  # Graph structure [N, N]
        flow_x = data["flow_x"].to(device)  # Input data [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # Reshape input data

        output_1_chb, pearson_1 = self.conv1(flow_x, graph_data, Y_data, Dist_data)
        output_1 = self.act(output_1_chb)

        output_2_chb = self.conv2(output_1, graph_data, None, None)
        output_2 = self.act(output_2_chb)

        output_2_tran = output_2.transpose(1, 2)

        output_3 = self.act(self.linear3(output_2_tran))

        pearson_sigmoid = self.act_pear(pearson_1)

        return output_3, pearson_sigmoid

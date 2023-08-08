# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    Visualization results and Evaluation
# ----------------------------------
from data_processing import LoadData
import matplotlib.pyplot as plt
import numpy as np
import h5py


class Evaluation(object):
    def __init__(self):
        pass

    @staticmethod
    def mae_(target, output):
        """
        Calculate Mean Absolute Error (MAE) between target and output.

        :param target: Ground truth target values.
        :param output: Predicted output values.
        :return: Calculated MAE.
        """
        return np.mean(np.abs(target - output))

    @staticmethod
    def mape_(target, output):
        """
        Calculate Mean Absolute Percentage Error (MAPE) between target and output.

        :param target: Ground truth target values.
        :param output: Predicted output values.
        :return: Calculated MAPE.
        """
        return np.mean(np.abs(target - output) / (target + 5))  # 5 is a small number to avoid zero-division.

    @staticmethod
    def rmse_(target, output):
        """
        Calculate Root Mean Squared Error (RMSE) between target and output.

        :param target: Ground truth target values.
        :param output: Predicted output values.
        :return: Calculated RMSE.
        """
        return np.sqrt(np.mean(np.power(target - output, 2)))

    @staticmethod
    def total(target, output):
        """
        Calculate a combination of evaluation metrics including MAE, MAPE, and RMSE.

        :param target: Ground truth target values.
        :param output: Predicted output values.
        :return: Calculated MAE, MAPE, and RMSE.
        """
        mae = Evaluation.mae_(target, output)
        mape = Evaluation.mape_(target, output)
        rmse = Evaluation.rmse_(target, output)

        return mae, mape, rmse


def visualize_result(h5_file, nodes_id, time_se):
    # Open the HDF5 file
    file_obj = h5py.File(h5_file, "r")

    # Extract the prediction and target data from the HDF5 file
    prediction = file_obj["predict"][:][:, :, 0]
    target = file_obj["target"][:][:, :, 0]

    # Close the HDF5 file
    file_obj.close()

    # Extract the prediction data for the specified node and time range
    plot_prediction = (prediction[nodes_id][time_se[0] : time_se[1]]).reshape(-1, 1)

    # Extract the target data for the specified node and time range
    plot_target = (target[nodes_id][time_se[0] : time_se[1]]).reshape(-1, 1)

    return plot_prediction, plot_target

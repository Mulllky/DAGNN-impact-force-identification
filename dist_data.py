# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    Sensor data truncated
# ----------------------------------
import numpy as np
import scipy.io as sio


def dist_data(index_i, index_j, Mode, history_length):
    """
    Prepare truncated data from sensor readings.

    :param index_i: List[int], list of i indices.
    :param index_j: List[int], list of j indices.
    :param Mode: str, mode for selecting data.
    :param history_length: int, length of historical data.
    :return: data_output: numpy array, prepared sensor data.
             data_output_force: numpy array, prepared force data.
    """

    # Path to the data directory
    path = './SensorData/Experiment/FourSensors/'

    data_read = []

    # Load data from files
    for i in range(len(index_i)):
        for j in range(len(index_j)):
            data_index = sio.loadmat(
                path + str(Mode) + '/' + 'Data_' + str(index_i[i]) + '_' + str(index_j[j]) + '.mat'
            )['Data']
            # Calculate the mean of sensor data excluding the first column
            data_index[:, 0] = np.mean(data_index[:, 1:], axis=1)
            data_read.append(data_index)

    data_read = np.array(data_read)

    index = 370  # Index of the truncated start force mark

    data = []

    # Extract a specific range of data around the index
    for _ in range(data_read.shape[0]):
        data = data_read[:, int(index - history_length / 2) : int(index + history_length / 2), :].astype(np.float32)

    # Transpose data dimensions
    data_output = data.transpose(0, 2, 1)

    # Calculate virtual force and replace the first time step with the force value
    force_virtual = np.sum(data_output[:, 1:, :], axis=1) / 40000
    data_output[:, 0, :] = force_virtual

    return data_output

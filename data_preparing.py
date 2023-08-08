# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    Sensor data preparation
# ----------------------------------
import numpy as np
import scipy.io as sio


def data_preparing(index_i, index_j, Mode, history_length):
    """
    Prepare data from sensor readings.

    :param index_i: List[int], list of i indices.
    :param index_j: List[int], list of j indices.
    :param Mode: str, mode for selecting data.
    :param history_length: int, length of historical data.
    :return: data_output: numpy array, prepared sensor data.
             data_output_force: numpy array, prepared force data.
    """

    path = './SensorData/Experiment/SixSensors/'

    data_read = []

    # Load data for given indices
    for i in range(len(index_i)):
        for j in range(len(index_j)):
            data_index = sio.loadmat(
                path + str(Mode) + '/' + 'Data_' + str(index_i[i]) + '_' + str(index_j[j]) + '.mat'
            )['Data']
            data_read.append(data_index)

    data_read = np.array(data_read)
    data = np.reshape(data_read, (-1, data_read.shape[2]))

    data_output = np.zeros((data.shape[1], data.shape[0], 1), dtype='float64')
    data_output_force = np.zeros((data.shape[1], data.shape[0], 1), dtype='float64')

    # Reshape and organize data
    for i in range(data.shape[1]):
        data_output[i, :, :] = data[:, i].reshape(-1, 1)
        data_output_force[i, :, :] = data[:, 0].reshape(-1, 1)

    # Calculate virtual force from data
    force_virtual = np.sum(data_output[1:, :, :], axis=0) / 40000
    data_output[0, :, :] = force_virtual.reshape(-1, 1)

    return data_output, data_output_force

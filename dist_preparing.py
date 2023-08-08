# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    Sensor distance data preparation
# ----------------------------------
import numpy as np
import scipy.io as sio


def dist_preparing(index_i, index_j, Mode, history_length):
    """
    Prepare data from sensor readings.

    :param index_i: List[int], list of i indices.
    :param index_j: List[int], list of j indices.
    :param Mode: str, mode for selecting data.
    :param history_length: int, length of historical data.
    :return: data_output: numpy array, prepared sensor data.
             data_output_force: numpy array, prepared force data.
    """

    # Path to the distance data file
    path = './SensorData/Experiment/FourSensors/distance.mat'

    # Load training and testing distance data
    training_dist = sio.loadmat(path)['training_dist']
    testing_dist = sio.loadmat(path)['testing_dist']

    dist_read = []

    # Extract distance data based on index_i and Mode
    for i in range(len(index_i)):
        for _ in range(len(index_j)):
            if Mode == "train":
                dist_index = training_dist[int(index_i[i] - 1), :]
            elif Mode == "test":
                dist_index = testing_dist[int(index_i[i] - 1), :]
            else:
                print("Wrong Mode!")

            dist_read.append(dist_index)
    dist_sensor = np.array(dist_read).T
    dist_sensor.astype(np.float32)

    return dist_sensor

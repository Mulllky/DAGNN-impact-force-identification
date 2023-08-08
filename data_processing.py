# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    Sensor data processing
# ----------------------------------
import torch
import numpy as np
from torch.utils.data import Dataset
from data_preparing import data_preparing
from adjacent_matrix import get_adjacent_matrix


class LoadData(Dataset):
    def __init__(
        self,
        graph_data_path,
        num_nodes,
        history_length,
        train_mode,
        train_set_i,
        train_set_j,
        test_set_i,
        test_set_j,
        delta_t,
        each_group_time,
    ):
        self.graph_data_path = graph_data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.each_group_time = each_group_time

        self.history_length = history_length
        self.time_interval = delta_t
        self.one_day_length = 1 / self.time_interval

        self.train_time = int(len(train_set_i) * len(train_set_j) * self.each_group_time * self.one_day_length)
        self.test_time = int(len(test_set_i) * len(test_set_j) * self.each_group_time * self.one_day_length)

        self.graph = get_adjacent_matrix(Graph_file=graph_data_path, num_nodes=num_nodes, Graph_type="distance")

        self.train_data, self.train_force = data_preparing(
            train_set_i, train_set_j, Mode="train", history_length=self.history_length
        )
        self.history_data = np.zeros(
            (self.train_data.shape[0], self.history_length, self.train_data.shape[2]), dtype='float64'
        )
        self.test_data, self.test_force = data_preparing(
            test_set_i, test_set_j, Mode="test", history_length=self.history_length
        )
        self.force_data = np.concatenate(
            (self.history_data, self.train_force, self.history_data, self.test_force), axis=1
        )
        self.flow_data = np.concatenate((self.history_data, self.train_data, self.history_data, self.test_data), axis=1)

    def __len__(self):
        if self.train_mode == "train":
            return self.train_time  # Number of training samples = total training dataset length
        elif self.train_mode == "test":
            return (
                self.test_time
            )  # Each sample can be used for testing, number of testing samples = total testing dataset length
        else:
            raise ValueError(f"train mode: [{self.train_mode}] is not defined")

    def __getitem__(self, index):
        if self.train_mode == "train":
            index = index  # Training data starts from time 0, this is individual flow data, distinguishable from (x, y) samples
        elif self.train_mode == "test":
            index += self.train_time + self.history_length  # Add an offset for testing data
        else:
            raise ValueError(f"train mode: [{self.train_mode}] is not defined")

        data_x = LoadData.slice_data_x(self.flow_data, self.history_length, index, self.train_mode)  # This is sample x
        data_y = LoadData.slice_data_y(self.force_data, self.history_length, index, self.train_mode)  # This is sample y

        data_x = LoadData.to_tensor(data_x)  # Convert to tensor [N, H, D]
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)  # Convert to tensor, add an extra time dimension [N, 1, D]

        # Return as dictionary
        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data_x(data, history_length, index, train_mode):
        if train_mode not in ["train", "test"]:
            raise ValueError(f"train model {train_mode} is not defined")
        start_index = index  # Starting index is the time index itself, this is a closed interval
        end_index = index + history_length  # Ending index, this is an open interval
        return data[:, start_index:end_index]

    @staticmethod
    def slice_data_y(data, history_length, index, train_mode):
        if train_mode not in ["train", "test"]:
            raise ValueError(f"train model {train_mode} is not defined")
        end_index = index + history_length  # Ending index, this is an open interval
        return data[:, end_index]

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


if __name__ == '__main__':
    train_data = LoadData(
        graph_data_path="GraphData/GraphDAGNN.csv",
        num_nodes=5,
        train_set_i=[1, 2, 3],
        train_set_j=[1, 2, 3],
        test_set_i=[1, 2, 3],
        test_set_j=[1, 2, 3],
        history_length=4,
        delta_t=0.001,
        each_group_time=2,
        train_mode="train",
    )

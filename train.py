# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    DAGNN model training
# ----------------------------------
import h5py
import numpy as np
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import Evaluation
from DAGNN_model import DAGNN_Net
from data_processing import LoadData
from dist_data import dist_data
from dist_preparing import dist_preparing


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Initial Setup
    GraphData_path = "GraphData/GraphDAGNN.csv"  # Path to graph data
    Model_name = "Model_A"  # Model name for saving
    Model_save_PATH = f"./TorchModel/{Model_name}.pt"  # Model saving path
    Loss_save_PATH = f"./LossSave/{Model_name}.txt"  # Loss log file path
    Vali_Loss_save_PATH = f"./LossSave/{Model_name}-Vali.txt"  # Validation loss log file path

    # Learning rate log file path
    lr_Loss_save_PATH = f"./LossSave/{Model_name}-lr.txt"

    # Hyperparameters
    history_length = 32  # Historical time steps
    batch_size = 256  # Batch size
    hid_channel = 512  # Hidden channels in the model
    lr_ori = 1e-4  # Initial learning rate
    Epoch = 50  # Number of training epochs

    # Experiment parameters
    each_group_time = 2  # Time for each group
    train_set_i = list(range(1, 15))  # Train set indices i
    train_set_j = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Train set indices j
    test_set_i = list(range(1, 15))  # Test set indices i
    test_set_j = [18]  # Test set indices j
    num_sensor = 4  # Number of sensors
    num_force = 1  # Number of forces

    # Total number of nodes in the graph
    Graph_num_nodes = int(num_force + num_sensor)
    delta_t = 1e-4  # Time step

    # Calculate batch groups
    batch_group_train = round(int(len(train_set_i) * len(train_set_j) * each_group_time / delta_t / batch_size))
    batch_group_test = round(int(len(test_set_i) * len(test_set_j) * each_group_time / delta_t / batch_size))

    # Load train Dataset
    train_data = LoadData(
        graph_data_path=GraphData_path,
        num_nodes=Graph_num_nodes,
        train_set_i=train_set_i,
        train_set_j=train_set_j,
        test_set_i=test_set_i,
        test_set_j=test_set_j,
        history_length=history_length,
        delta_t=delta_t,
        each_group_time=each_group_time,
        train_mode="train",
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load and prepare distributed training data
    train_dist_data = torch.from_numpy(dist_data(train_set_i, train_set_j, Mode="train", history_length=history_length))
    train_dist = torch.from_numpy(dist_preparing(train_set_i, train_set_j, Mode="train", history_length=history_length))

    # Load test dataset
    test_data = LoadData(
        graph_data_path=GraphData_path,
        num_nodes=Graph_num_nodes,
        train_set_i=train_set_i,
        train_set_j=train_set_j,
        test_set_i=test_set_i,
        test_set_j=test_set_j,
        history_length=history_length,
        delta_t=delta_t,
        each_group_time=each_group_time,
        train_mode="test",
    )

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load and prepare distributed test data
    test_dist_data = torch.from_numpy(dist_data(test_set_i, test_set_j, Mode="test", history_length=history_length))
    test_dist = torch.from_numpy(dist_preparing(test_set_i, test_set_j, Mode="test", history_length=history_length))

    # Load Model
    model = DAGNN_Net(in_c=history_length, hid_c=hid_channel, out_c=1, num_node=Graph_num_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr_ori)

    cur_lr_list = []
    # To prevent the first loss from being too large
    total_loss_all = [1000000]

    # Train Model
    model.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()

        for data in train_loader:
            model.zero_grad()
            # Compute predictions and Pearson correlation
            # data = {"graph": [N, N], "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]}
            predict_value, pear = model(data, device, train_dist_data.to(device), train_dist.to(device))[0].to(
                torch.device("cpu")
            ), model(data, device, train_dist_data.to(device), train_dist.to(device))[1].to(torch.device("cpu"))
            target_value = data["flow_y"][:, 0, ::]
            loss = criterion(predict_value, target_value)
            loss_pear = torch.div(loss, pear)
            train_pear = pear
            epoch_loss += loss_pear.item()
            loss_pear.backward()
            optimizer.step()

        end_time = time.time()

        # Prevent overfitting
        with torch.no_grad():
            total_loss = 0.0
            for data in test_loader:
                predict_value, pear = model(data, device, test_dist_data.to(device), test_dist.to(device))[0].to(
                    torch.device("cpu")
                ), model(data, device, test_dist_data.to(device), test_dist.to(device))[1].to(torch.device("cpu"))
                target_value = data["flow_y"][:, 0, ::]
                loss = criterion(predict_value, target_value)
                loss_pear = torch.div(loss, pear)
                total_loss += loss_pear.item()

            if total_loss > total_loss_all[-1] * 1.1:
                lr_ori = lr_ori * 0.7

            total_loss_all.append(total_loss)

        if epoch >= 1:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_ori

        cur_lr = optimizer.param_groups[-1]["lr"]
        cur_lr_list.append(cur_lr)

        print(
            "Epoch: {:04d}, Train Loss: {:02.4f}, Test Loss: {:02.4f}, Time: {:02.2f} mins, Current lr: {:02.2e}, Pearson: {:02.4f}".format(
                epoch,
                100 * epoch_loss / batch_group_train,
                100 * total_loss / batch_group_test,
                (end_time - start_time) / 60,
                cur_lr,
                train_pear,
            )
        )

        # Log losses
        with open(Loss_save_PATH, "a+") as LossTxt:
            LossTxt.writelines("Loss: {:02.4f}".format(100 * epoch_loss / batch_group_train) + "\n")
        with open(Vali_Loss_save_PATH, "a+") as LossTxt:
            LossTxt.writelines("Loss: {:02.4f}".format(100 * total_loss / batch_group_train) + "\n")
        with open(lr_Loss_save_PATH, "a+") as LossTxt:
            LossTxt.writelines("Loss: {:02.8f}".format(cur_lr) + "\n")

    # Test Model
    model.eval()
    with torch.no_grad():
        MAE, MAPE, RMSE = [], [], []
        Target = np.zeros([1, 1, 1])
        Predict = np.zeros_like(Target)

        test_loss = 0.0
        for data in test_loader:
            predict_value, pear = model(data, device, test_dist_data.to(device), test_dist.to(device))[0].to(
                torch.device("cpu")
            ), model(data, device, test_dist_data.to(device), test_dist.to(device))[1].to(torch.device("cpu"))
            target_value = data["flow_y"][:, 0, ::]
            loss = criterion(predict_value, target_value)
            loss_pear = torch.div(loss, pear)
            test_loss += loss_pear.item()
            predict_value = predict_value.transpose(0, 1)
            target_value = (data["flow_y"][:, 0, ::]).transpose(0, 1)

            performance, data_to_save = compute_performance(predict_value, target_value, test_loader)

            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
            Target = np.concatenate([Target, data_to_save[1]], axis=1)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

    # Save Model
    torch.save(model, Model_save_PATH)

    # Compute Average Performance Metrics
    print(
        "Performance:  MAE: {:2.2f}   MAPE: {:2.2f}%   RMSE: {:2.2f}".format(
            np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)
        )
    )

    # Remove Padding and Save Results to HDF5 File
    Predict = np.delete(Predict, 0, axis=1)
    Target = np.delete(Target, 0, axis=1)

    result_file = "DAGNN_Net_result.h5"
    file_obj = h5py.File(result_file, "w")

    file_obj["predict"] = Predict  # [N, T, D]
    file_obj["target"] = Target  # [N, T, D]

    file_obj.close()

    print("ok")


def compute_performance(prediction, target, data):
    # Check if the data is in DataLoader format or Dataset format
    try:
        dataset = data.dataset  # If DataLoader, access its underlying dataset
    except:
        dataset = data  # If already in Dataset format, assign directly

    # Convert prediction and target to NumPy arrays
    prediction = prediction.numpy()
    target = target.numpy()

    # Calculate performance metrics: MAE, MAPE, RMSE
    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))

    # Store calculated performance metrics and recovered data
    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    # Return performance metrics and recovered data
    return performance, recovered_data


if __name__ == "__main__":
    main()

    print("ok")

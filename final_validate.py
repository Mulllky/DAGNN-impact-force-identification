# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    DAGNN model validation
# ----------------------------------
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_processing import LoadData
from utils import Evaluation
from utils import visualize_result
from dist_data import dist_data
from dist_preparing import dist_preparing


def main():
    # Load Model
    GraphData_path = "GraphData/GraphDAGNN.csv"  # Path to graph data
    Model_name = 'Model_A'  # Model name for saving
    Model_save_PATH = f'./TorchModel/{Model_name}.pt'  # Model saving path

    model = torch.load(Model_save_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    # Hyperparameters
    history_length = 32  # Historical time steps
    batch_size = 256  # Batch size

    # Experiment parameters
    each_group_time = 2  # Time for each group
    train_set_i = list(range(1, 15))  # Train set indices i
    train_set_j = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Train set indices j
    validate_set_i = list(range(1, 50))  # Validate set indices i
    validate_set_j = [1]  # Validate set indices j
    num_sensor = 4  # Number of sensors
    num_force = 1  # Number of forces
    Graph_num_nodes = int(num_force + num_sensor)  # Total number of nodes in the graph
    delta_t = 1e-4  # Time step

    # Load validate dataset
    validate_data = LoadData(
        graph_data_path=GraphData_path,
        num_nodes=Graph_num_nodes,
        train_set_i=train_set_i,
        train_set_j=train_set_j,
        test_set_i=validate_set_i,
        test_set_j=validate_set_j,
        history_length=history_length,
        delta_t=delta_t,
        each_group_time=each_group_time,
        train_mode="test",
    )

    validate_loader = DataLoader(validate_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load and prepare distributed validate data
    validate_dist_data = torch.from_numpy(
        dist_data(validate_set_i, validate_set_j, Mode="test", history_length=history_length)
    )
    validate_dist = torch.from_numpy(
        dist_preparing(validate_set_i, validate_set_j, Mode="test", history_length=history_length)
    )

    def compute_performance(prediction, target, data):
        # Check if the data is in DataLoader format or Dataset format
        try:
            dataset = data.dataset  # If DataLoader, access its underlying dataset
        except Exception:
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

    # Validate Model
    model.eval()
    with torch.no_grad():
        MAE, MAPE, RMSE = [], [], []
        Target = np.zeros([1, 1, 1])
        Predict = np.zeros_like(Target)

        test_loss = 0.0
        for data in validate_loader:
            predict_value, pear = model(data, device, validate_dist_data.to(device), validate_dist.to(device))[0].to(
                torch.device("cpu")
            ), model(data, device, validate_dist_data.to(device), validate_dist.to(device))[1].to(torch.device("cpu"))
            target_value = data["flow_y"][:, 0, ::]
            loss = criterion(predict_value, target_value)
            loss_pear = torch.div(loss, pear)
            test_loss += loss_pear.item()
            predict_value = predict_value.transpose(0, 1)
            target_value = (data["flow_y"][:, 0, ::]).transpose(0, 1)

            performance, data_to_save = compute_performance(predict_value, target_value, validate_loader)

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

    # Visualize Weight Values
    # Extract and visualize weight values from the model's first convolutional layer
    I_weight = ((model.conv1.weight[0, ::]).squeeze(0)).to(torch.device("cpu")).detach().numpy()
    L_weight = ((model.conv1.weight[1, ::]).squeeze(0)).to(torch.device("cpu")).detach().numpy()

    # Visualize Results
    # Perform visualization of results
    # In the following Evaluation() class, this corresponds to visualizing the results of the DAGNN_Net algorithm
    test_time = round(len(validate_set_i) * len(validate_set_j) * each_group_time)
    t_data = np.reshape(np.arange(0, test_time, delta_t), (-1, 1))

    out_target, out_prediction = np.zeros_like(t_data), np.zeros_like(t_data)

    time_se_end = int(t_data.shape[0])

    # Visualize the results using the visualize_result() function
    data_prediction, data_target = visualize_result(h5_file="DAGNN_Net_result.h5", nodes_id=0, time_se=[0, time_se_end])
    out_target += data_target
    out_prediction += data_prediction

    # Prepare data for plotting
    plot_target, plot_prediction = out_target, out_prediction


if __name__ == '__main__':
    main()

    print('ok')

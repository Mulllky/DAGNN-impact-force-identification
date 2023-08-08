# -*- encoding: utf-8 -*-
# Date:    2023/8/8 15:32
# Author:  Chun Huang
# Github:  https://github.com/Mulllky
# Desc:    Adjacency matrix calculation
# ----------------------------------
import csv
import numpy as np


def get_adjacent_matrix(Graph_file, num_nodes, Graph_type="connect"):
    """
    Generate an adjacency matrix based on Graph.

    :param Graph_file (str): Path to the CSV file containing Graph information.
    :param num_nodes (int): Total number of nodes in the graph.
    :param Graph_type (str, optional): Type of graph ("connect" for connectivity, "distance" for weighted by distance). Defaults to "connect".
    :return: A: numpy array, The adjacency matrix.
    """
    A = np.zeros([num_nodes, num_nodes])  # Initialize an all-zero adjacency matrix

    with open(Graph_file, "r") as Graph:
        Graph.readline()  # Skip the header line.
        Graph_reader = csv.reader(Graph)  # Read the .csv file.
        for node_item in Graph_reader:  # Process each line as an node_item
            if len(node_item) != 3:  # If the line doesn't have 3 elements, skip it
                continue
            i, j, distance = int(node_item[0]), int(node_item[1]), float(node_item[2])

            if Graph_type == "connect":
                # Set the adjacency values to 1 for connectivity graph
                A[i, j], A[j, i] = 1.0, 1.0
            elif Graph_type == "distance":
                # Calculate and set weighted adjacency values for distance graph
                A[i, j] = 1.0 / distance
                A[j, i] = 1.0 / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")

    return A

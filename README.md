# Impact Force Reconstruction and Localization using Distance-assisted Graph Neural Network

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

## Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Files and Scripts](#files-and-scripts)
- [References](#references)
- [Contact](#contact)

## Abstract

In this paper, we introduce a novel impact force identification approach known as Distance-assisted Graph Neural Network (DAGNN). This method is designed to simultaneously achieve accurate force localization and force history reconstruction. DAGNN utilizes a graph-based supervised learning framework, where the sensors are treated as nodes and their interconnections as edges. We mathematically characterize the relationships between nodes and edges using adjacency and degree matrices, forming the foundation of the Graph Neural Network (GNN). To enhance learning and enable precise force localization, we incorporate physical distances between the impact force position and sensors as physics-informed constraints during the training process. An experimental validation on a metal plate demonstrates the efficacy and practicality of DAGNN. By collecting force and response signals through a sparse grid arrangement on the plate, we create a dataset for training the DAGNN model. The results reveal that DAGNN outperforms conventional GNN models, such as Graph Convolutional Networks (GCN) and Artificial Neural Network (ANN) models, in terms of time history reconstruction accuracy. Moreover, DAGNN maintains high accuracy across various impact locations due to the infused distance information. This approach enables the construction of a straightforward relation between impact and sensor distances, achieving accurate impact localization.

## Features
- Simultaneous impact force localization and force history reconstruction.
- Graph-based approach using Distance-assisted Graph Neural Network.
- Integration of physical distances to enhance model learning and localization accuracy.

## Dependencies

- audtorch==0.6.4
- h5py==3.7.0
- matplotlib==3.6.2
- numpy==1.21.5
- scipy==1.10.0
- torch==1.12.1

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/Mulllky/DAGNN-impact-force-identification.git
    cd DAGNN-impact-force-identification
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Files and Scripts

- **GraphData:** Contains graph-related information.
- **LossSave:** Stores outputs related to loss functions.
- **Output:** Stores outputs generated by the model.
- **SensorData:** Contains response and load samples.
- **TorchModel:** Stores trained model files.

- **adjacent_matrix.py:** Converts GraphData's GraphDAGNN.csv to an adjacency matrix.
- **DAGNN_model.py:** Implements internal computations of the DAGNN model.
- **data_preparing.py:** Reads samples from SensorData.
- **data_processing.py:** Processes sample data.
- **dist_data.py:** Extracts peak values from responses.
- **dist_preparing.py:** Reads distance information.
- **final_validate.py:** Validates trained models.
- **pearson_calculation.py:** Computes Pearson coefficients.
- **train.py:** Trains the model.
- **utils.py:** Evaluates trained model outputs.

## References

C. Huang, C. Tao, H. Ji, J. Qiu, "Impact force reconstruction and localization using Distance-assisted Graph Neural Network," Mechanical Systems and Signal Processing. 200 (2023) 110606. [DOI](https://doi.org/10.1016/j.ymssp.2023.110606)

## Contact

For any inquiries, please contact [huangchun@nuaa.edu.cn](mailto:huangchun@nuaa.edu.cn).


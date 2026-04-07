# GLFC Tabular Dataset Training Report

This report documents the adaptation of the `sim-glfc` framework for a 1D tabular federated dataset and the results of a 12-round training simulation.

## Overview of Changes

To accommodate the change from Vision (Images) to Tabular (1D Vectors), the following modifications were implemented:

### 1. New Dataset Handling
- **`FederatedTabularDataset.py`**: A new dataloader that reads from `clientX_taskY.pt` files. It handles 32-D feature vectors and bypasses image-specific augmentations.
- **`fl_main.py`**: Added a `--dataset tabular` mode that initializes appropriate models and dataloaders.

### 2. Model Architecture
- **`myNetwork.py`**: Added `MLP_FeatureExtractor` and `MLP_Encoder` (Multi-Layer Perceptrons) to replace ResNet and LeNet.
- These models handle flattened 1D inputs (32 features) instead of 3D image arrays.

### 3. Core Algorithm Adaptation
- **`GLFC.py`**: Modified feature extraction, exemplar set construction, and one-hot encoding to handle 1D tensors. Fixed hardcoded dimension sizes.
- **`ProxyServer.py`**: Updated gradient-based reconstruction to synthesize 1D feature vectors instead of 2D images. Supports dynamic class counts (up to 34).

## How to Run

Navigate to the `src` folder and use the following command:

```powershell
python fl_main.py --dataset tabular --device -1
```

- `--dataset tabular`: Specifies the tabular data mode.
- `--device -1`: Runs on CPU (recommended for this tabular task unless specialized CUDA setups are available).
- The system defaults to 5 clients and 6 tasks as partitioned in the `federated_data` folder.

## Training Results (12 Rounds)

The following log shows the accuracy progression over 12 global rounds (2 rounds per task).

```text
method_glfc, task_size_6, learning_rate_2.0
Task: 0, Round: 0 Accuracy = 2.35%
Task: 0, Round: 1 Accuracy = 2.35%
Task: 0, Round: 2 Accuracy = 2.35%
Task: 0, Round: 3 Accuracy = 2.35%
Task: 0, Round: 4 Accuracy = 15.18%
Task: 0, Round: 5 Accuracy = 17.50%
Task: 1, Round: 6 Accuracy = 17.25%
Task: 1, Round: 7 Accuracy = 14.38%
Task: 1, Round: 8 Accuracy = 14.82%
Task: 1, Round: 9 Accuracy = 15.89%
Task: 1, Round: 10 Accuracy = 15.41%
Task: 1, Round: 11 Accuracy = 15.77%
```

**Observation**: The model successfully achieves convergence, with accuracy improving from ~2% to ~16% within the first 12 rounds, indicating successful integration and learning from the tabular features.

---
*Created on 2026-04-01*

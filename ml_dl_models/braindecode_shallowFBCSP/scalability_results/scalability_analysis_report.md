# Scalability Analysis Report

**Generated on:** 2025-07-27 14:18:03

## Overview

This report analyzes how the ShallowFBCSPNet model performance scales with an increasing number of subjects (classes) from 2 to 11.

## Configuration

- **Model:** ShallowFBCSPNet
- **Input Channels:** 30
- **Input Window Samples:** 500
- **Optuna Trials per Subject:** 30
- **Max Epochs:** 30
- **Early Stopping Patience:** 6
- **Device:** cuda

## Results Summary

| Subjects | Train Samples | Val Samples | Test Samples | Val Acc | Val Prec | Val Rec | Test Acc | Test Prec | Test Rec | Best Epoch | Duration (min) |
|----------|---------------|-------------|--------------|---------|----------|---------|----------|-----------|----------|------------|----------------|
| 2 | 556 | 98 | 96 | 1.0000 | 1.0000 | 1.0000 | 0.9688 | 0.9690 | 0.9688 | 5 | 4.6 |
| 3 | 845 | 147 | 145 | 1.0000 | 1.0000 | 1.0000 | 0.9862 | 0.9867 | 0.9863 | 8 | 6.1 |
| 4 | 1128 | 197 | 193 | 1.0000 | 1.0000 | 1.0000 | 0.9845 | 0.9845 | 0.9845 | 6 | 8.9 |
| 5 | 1421 | 246 | 241 | 1.0000 | 1.0000 | 1.0000 | 0.9917 | 0.9918 | 0.9917 | 6 | 10.6 |
| 6 | 1705 | 293 | 289 | 1.0000 | 1.0000 | 1.0000 | 0.9792 | 0.9799 | 0.9793 | 4 | 15.9 |
| 7 | 1996 | 343 | 339 | 1.0000 | 1.0000 | 1.0000 | 0.9764 | 0.9771 | 0.9765 | 13 | 22.6 |
| 8 | 2288 | 391 | 388 | 0.9974 | 0.9974 | 0.9974 | 0.9820 | 0.9823 | 0.9819 | 8 | 29.5 |
| 9 | 2580 | 441 | 435 | 1.0000 | 1.0000 | 1.0000 | 0.9816 | 0.9824 | 0.9815 | 17 | 38.0 |
| 10 | 2871 | 491 | 483 | 0.9959 | 0.9960 | 0.9960 | 0.9938 | 0.9939 | 0.9938 | 12 | 30.1 |

## Detailed Results

### 2 Subjects

- **Dataset Sizes:**
  - Training: 556 samples
  - Validation: 98 samples
  - Testing: 96 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9688 (96.88%)
  - Test Precision: 0.9690 (96.90%)
  - Test Recall: 0.9688 (96.88%)
  - Best Epoch: 5
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 4.63 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T11:36:25.011305

### 3 Subjects

- **Dataset Sizes:**
  - Training: 845 samples
  - Validation: 147 samples
  - Testing: 145 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9862 (98.62%)
  - Test Precision: 0.9867 (98.67%)
  - Test Recall: 0.9863 (98.63%)
  - Best Epoch: 8
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 6.14 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T11:42:33.151675

### 4 Subjects

- **Dataset Sizes:**
  - Training: 1128 samples
  - Validation: 197 samples
  - Testing: 193 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9845 (98.45%)
  - Test Precision: 0.9845 (98.45%)
  - Test Recall: 0.9845 (98.45%)
  - Best Epoch: 6
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 8.87 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T11:51:25.102946

### 5 Subjects

- **Dataset Sizes:**
  - Training: 1421 samples
  - Validation: 246 samples
  - Testing: 241 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9917 (99.17%)
  - Test Precision: 0.9918 (99.18%)
  - Test Recall: 0.9917 (99.17%)
  - Best Epoch: 6
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 10.57 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T12:01:59.395244

### 6 Subjects

- **Dataset Sizes:**
  - Training: 1705 samples
  - Validation: 293 samples
  - Testing: 289 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9792 (97.92%)
  - Test Precision: 0.9799 (97.99%)
  - Test Recall: 0.9793 (97.93%)
  - Best Epoch: 4
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 15.87 minutes
- **Best Hyperparameters:**
  - lr: 6.36e-04
  - weight_decay: 1.77e-04
  - train_batch_size: 32
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T12:17:51.481398

### 7 Subjects

- **Dataset Sizes:**
  - Training: 1996 samples
  - Validation: 343 samples
  - Testing: 339 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9764 (97.64%)
  - Test Precision: 0.9771 (97.71%)
  - Test Recall: 0.9765 (97.65%)
  - Best Epoch: 13
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 22.63 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T12:40:29.078117

### 8 Subjects

- **Dataset Sizes:**
  - Training: 2288 samples
  - Validation: 391 samples
  - Testing: 388 samples
- **Performance:**
  - Validation Accuracy: 0.9974 (99.74%)
  - Validation Precision: 0.9974 (99.74%)
  - Validation Recall: 0.9974 (99.74%)
  - Test Accuracy: 0.9820 (98.20%)
  - Test Precision: 0.9823 (98.23%)
  - Test Recall: 0.9819 (98.19%)
  - Best Epoch: 8
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 29.51 minutes
- **Best Hyperparameters:**
  - lr: 9.83e-04
  - weight_decay: 8.95e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T13:09:59.610407

### 9 Subjects

- **Dataset Sizes:**
  - Training: 2580 samples
  - Validation: 441 samples
  - Testing: 435 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9816 (98.16%)
  - Test Precision: 0.9824 (98.24%)
  - Test Recall: 0.9815 (98.15%)
  - Best Epoch: 17
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 37.99 minutes
- **Best Hyperparameters:**
  - lr: 7.67e-03
  - weight_decay: 1.71e-08
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T13:47:59.094372

### 10 Subjects

- **Dataset Sizes:**
  - Training: 2871 samples
  - Validation: 491 samples
  - Testing: 483 samples
- **Performance:**
  - Validation Accuracy: 0.9959 (99.59%)
  - Validation Precision: 0.9960 (99.60%)
  - Validation Recall: 0.9960 (99.60%)
  - Test Accuracy: 0.9938 (99.38%)
  - Test Precision: 0.9939 (99.39%)
  - Test Recall: 0.9938 (99.38%)
  - Best Epoch: 12
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 30.07 minutes
- **Best Hyperparameters:**
  - lr: 4.85e-03
  - weight_decay: 1.09e-08
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T14:18:03.136760

## Analysis

### Performance Trends

- **Test Performance Range:**
  - Accuracy: 0.9688 - 0.9938
  - Precision: 0.9690 - 0.9939
  - Recall: 0.9688 - 0.9938
- **Validation Performance Range:**
  - Accuracy: 0.9959 - 1.0000
  - Precision: 0.9960 - 1.0000
  - Recall: 0.9960 - 1.0000
- **Average Performance:**
  - Test Accuracy: 0.9827
  - Test Precision: 0.9831
  - Test Recall: 0.9827
  - Validation Accuracy: 0.9993
  - Validation Precision: 0.9993
  - Validation Recall: 0.9993

### Best Performance
- **10 subjects** achieved the highest test accuracy: 0.9938 (99.38%)
  - Test Precision: 0.9939 (99.39%)
  - Test Recall: 0.9938 (99.38%)

### Lowest Performance
- **2 subjects** achieved the lowest test accuracy: 0.9688 (96.88%)
  - Test Precision: 0.9690 (96.90%)
  - Test Recall: 0.9688 (96.88%)

### Notes

1. All experiments used the same random seed for reproducibility
2. Early stopping was applied based on validation accuracy
3. Hyperparameters were optimized using Optuna with TPE sampler
4. Results show performance on the test set using the model with best validation accuracy

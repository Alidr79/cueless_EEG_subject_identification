# Scalability Analysis Report

**Generated on:** 2025-07-27 21:38:18

## Overview

This report analyzes how the EEGNetv1 model performance scales with an increasing number of subjects (classes) from 2 to 11.

## Configuration

- **Model:** EEGNetv1
- **Input Channels:** 30
- **Input Window Samples:** 500
- **Optuna Trials per Subject:** 30
- **Max Epochs:** 30
- **Early Stopping Patience:** 6
- **Device:** cuda

## Results Summary

| Subjects | Train Samples | Val Samples | Test Samples | Val Acc | Val Prec | Val Rec | Test Acc | Test Prec | Test Rec | Best Epoch | Duration (min) |
|----------|---------------|-------------|--------------|---------|----------|---------|----------|-----------|----------|------------|----------------|
| 2 | 556 | 98 | 96 | 0.9898 | 0.9898 | 0.9900 | 0.9896 | 0.9898 | 0.9896 | 14 | 8.5 |
| 3 | 845 | 147 | 145 | 1.0000 | 1.0000 | 1.0000 | 0.9862 | 0.9867 | 0.9861 | 12 | 14.3 |
| 4 | 1128 | 197 | 193 | 0.9949 | 0.9950 | 0.9950 | 0.9793 | 0.9798 | 0.9794 | 10 | 21.6 |
| 5 | 1421 | 246 | 241 | 0.9431 | 0.9487 | 0.9429 | 0.8755 | 0.9005 | 0.8754 | 13 | 27.7 |
| 6 | 1705 | 293 | 289 | 0.9454 | 0.9542 | 0.9452 | 0.9100 | 0.9188 | 0.9102 | 18 | 29.5 |
| 7 | 1996 | 343 | 339 | 0.9592 | 0.9621 | 0.9591 | 0.8850 | 0.9040 | 0.8853 | 16 | 30.3 |
| 8 | 2288 | 391 | 388 | 0.9258 | 0.9376 | 0.9257 | 0.8763 | 0.9003 | 0.8753 | 22 | 37.3 |
| 9 | 2580 | 441 | 435 | 0.9093 | 0.9170 | 0.9091 | 0.7839 | 0.8384 | 0.7842 | 8 | 39.4 |
| 10 | 2871 | 491 | 483 | 0.8880 | 0.9097 | 0.8879 | 0.7702 | 0.8363 | 0.7690 | 6 | 54.0 |

## Detailed Results

### 2 Subjects

- **Dataset Sizes:**
  - Training: 556 samples
  - Validation: 98 samples
  - Testing: 96 samples
- **Performance:**
  - Validation Accuracy: 0.9898 (98.98%)
  - Validation Precision: 0.9898 (98.98%)
  - Validation Recall: 0.9900 (99.00%)
  - Test Accuracy: 0.9896 (98.96%)
  - Test Precision: 0.9898 (98.98%)
  - Test Recall: 0.9896 (98.96%)
  - Best Epoch: 14
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 8.47 minutes
- **Best Hyperparameters:**
  - lr: 6.80e-03
  - weight_decay: 1.09e-08
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T17:24:17.051331

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
  - Test Recall: 0.9861 (98.61%)
  - Best Epoch: 12
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 14.26 minutes
- **Best Hyperparameters:**
  - lr: 9.29e-03
  - weight_decay: 8.11e-07
  - train_batch_size: 32
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T17:38:32.456666

### 4 Subjects

- **Dataset Sizes:**
  - Training: 1128 samples
  - Validation: 197 samples
  - Testing: 193 samples
- **Performance:**
  - Validation Accuracy: 0.9949 (99.49%)
  - Validation Precision: 0.9950 (99.50%)
  - Validation Recall: 0.9950 (99.50%)
  - Test Accuracy: 0.9793 (97.93%)
  - Test Precision: 0.9798 (97.98%)
  - Test Recall: 0.9794 (97.94%)
  - Best Epoch: 10
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 21.60 minutes
- **Best Hyperparameters:**
  - lr: 9.93e-03
  - weight_decay: 7.57e-08
  - train_batch_size: 32
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T18:00:08.524754

### 5 Subjects

- **Dataset Sizes:**
  - Training: 1421 samples
  - Validation: 246 samples
  - Testing: 241 samples
- **Performance:**
  - Validation Accuracy: 0.9431 (94.31%)
  - Validation Precision: 0.9487 (94.87%)
  - Validation Recall: 0.9429 (94.29%)
  - Test Accuracy: 0.8755 (87.55%)
  - Test Precision: 0.9005 (90.05%)
  - Test Recall: 0.8754 (87.54%)
  - Best Epoch: 13
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 27.71 minutes
- **Best Hyperparameters:**
  - lr: 6.80e-03
  - weight_decay: 1.09e-08
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T18:27:51.438461

### 6 Subjects

- **Dataset Sizes:**
  - Training: 1705 samples
  - Validation: 293 samples
  - Testing: 289 samples
- **Performance:**
  - Validation Accuracy: 0.9454 (94.54%)
  - Validation Precision: 0.9542 (95.42%)
  - Validation Recall: 0.9452 (94.52%)
  - Test Accuracy: 0.9100 (91.00%)
  - Test Precision: 0.9188 (91.88%)
  - Test Recall: 0.9102 (91.02%)
  - Best Epoch: 18
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 29.50 minutes
- **Best Hyperparameters:**
  - lr: 4.69e-03
  - weight_decay: 1.09e-08
  - train_batch_size: 32
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T18:57:21.489392

### 7 Subjects

- **Dataset Sizes:**
  - Training: 1996 samples
  - Validation: 343 samples
  - Testing: 339 samples
- **Performance:**
  - Validation Accuracy: 0.9592 (95.92%)
  - Validation Precision: 0.9621 (96.21%)
  - Validation Recall: 0.9591 (95.91%)
  - Test Accuracy: 0.8850 (88.50%)
  - Test Precision: 0.9040 (90.40%)
  - Test Recall: 0.8853 (88.53%)
  - Best Epoch: 16
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 30.30 minutes
- **Best Hyperparameters:**
  - lr: 1.89e-03
  - weight_decay: 1.12e-06
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T19:27:39.571132

### 8 Subjects

- **Dataset Sizes:**
  - Training: 2288 samples
  - Validation: 391 samples
  - Testing: 388 samples
- **Performance:**
  - Validation Accuracy: 0.9258 (92.58%)
  - Validation Precision: 0.9376 (93.76%)
  - Validation Recall: 0.9257 (92.57%)
  - Test Accuracy: 0.8763 (87.63%)
  - Test Precision: 0.9003 (90.03%)
  - Test Recall: 0.8753 (87.53%)
  - Best Epoch: 22
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 37.32 minutes
- **Best Hyperparameters:**
  - lr: 3.41e-03
  - weight_decay: 1.42e-06
  - train_batch_size: 32
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T20:04:58.486740

### 9 Subjects

- **Dataset Sizes:**
  - Training: 2580 samples
  - Validation: 441 samples
  - Testing: 435 samples
- **Performance:**
  - Validation Accuracy: 0.9093 (90.93%)
  - Validation Precision: 0.9170 (91.70%)
  - Validation Recall: 0.9091 (90.91%)
  - Test Accuracy: 0.7839 (78.39%)
  - Test Precision: 0.8384 (83.84%)
  - Test Recall: 0.7842 (78.42%)
  - Best Epoch: 8
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 39.37 minutes
- **Best Hyperparameters:**
  - lr: 1.67e-03
  - weight_decay: 2.02e-06
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T20:44:20.550657

### 10 Subjects

- **Dataset Sizes:**
  - Training: 2871 samples
  - Validation: 491 samples
  - Testing: 483 samples
- **Performance:**
  - Validation Accuracy: 0.8880 (88.80%)
  - Validation Precision: 0.9097 (90.97%)
  - Validation Recall: 0.8879 (88.79%)
  - Test Accuracy: 0.7702 (77.02%)
  - Test Precision: 0.8363 (83.63%)
  - Test Recall: 0.7690 (76.90%)
  - Best Epoch: 6
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 53.97 minutes
- **Best Hyperparameters:**
  - lr: 2.74e-03
  - weight_decay: 4.55e-07
  - train_batch_size: 32
  - val_batch_size: 32
- **Timestamp:** 2025-07-27T21:38:18.982866

## Analysis

### Performance Trends

- **Test Performance Range:**
  - Accuracy: 0.7702 - 0.9896
  - Precision: 0.8363 - 0.9898
  - Recall: 0.7690 - 0.9896
- **Validation Performance Range:**
  - Accuracy: 0.8880 - 1.0000
  - Precision: 0.9097 - 1.0000
  - Recall: 0.8879 - 1.0000
- **Average Performance:**
  - Test Accuracy: 0.8951
  - Test Precision: 0.9172
  - Test Recall: 0.8949
  - Validation Accuracy: 0.9506
  - Validation Precision: 0.9571
  - Validation Recall: 0.9505

### Best Performance
- **2 subjects** achieved the highest test accuracy: 0.9896 (98.96%)
  - Test Precision: 0.9898 (98.98%)
  - Test Recall: 0.9896 (98.96%)

### Lowest Performance
- **10 subjects** achieved the lowest test accuracy: 0.7702 (77.02%)
  - Test Precision: 0.8363 (83.63%)
  - Test Recall: 0.7690 (76.90%)

### Notes

1. All experiments used the same random seed for reproducibility
2. Early stopping was applied based on validation accuracy
3. Hyperparameters were optimized using Optuna with TPE sampler
4. Results show performance on the test set using the model with best validation accuracy

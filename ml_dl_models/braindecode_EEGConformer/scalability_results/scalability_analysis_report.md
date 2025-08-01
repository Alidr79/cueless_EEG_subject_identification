# Scalability Analysis Report

**Generated on:** 2025-07-31 17:59:47

## Overview

This report analyzes how the EEGConformer model performance scales with an increasing number of subjects (classes) from 2 to 11.

## Configuration

- **Model:** EEGConformer
- **Input Channels:** 30
- **Input Window Samples:** 500
- **Optuna Trials per Subject:** 30
- **Max Epochs:** 30
- **Early Stopping Patience:** 6
- **Device:** cuda

## Results Summary

| Subjects | Train Samples | Val Samples | Test Samples | Val Acc | Val Prec | Val Rec | Test Acc | Test Prec | Test Rec | Best Epoch | Duration (min) |
|----------|---------------|-------------|--------------|---------|----------|---------|----------|-----------|----------|------------|----------------|
| 2 | 556 | 98 | 96 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3 | 8.0 |
| 3 | 845 | 147 | 145 | 1.0000 | 1.0000 | 1.0000 | 0.9862 | 0.9863 | 0.9863 | 5 | 11.4 |
| 4 | 1128 | 197 | 193 | 1.0000 | 1.0000 | 1.0000 | 0.9845 | 0.9853 | 0.9847 | 5 | 13.5 |
| 5 | 1421 | 246 | 241 | 1.0000 | 1.0000 | 1.0000 | 0.9751 | 0.9752 | 0.9753 | 12 | 23.7 |
| 6 | 1705 | 293 | 289 | 1.0000 | 1.0000 | 1.0000 | 0.9792 | 0.9801 | 0.9795 | 8 | 28.0 |
| 7 | 1996 | 343 | 339 | 1.0000 | 1.0000 | 1.0000 | 0.9587 | 0.9613 | 0.9591 | 11 | 26.4 |
| 8 | 2288 | 391 | 388 | 0.9974 | 0.9975 | 0.9974 | 0.9897 | 0.9901 | 0.9898 | 23 | 52.2 |
| 9 | 2580 | 441 | 435 | 1.0000 | 1.0000 | 1.0000 | 0.9632 | 0.9651 | 0.9632 | 20 | 54.1 |
| 10 | 2871 | 491 | 483 | 1.0000 | 1.0000 | 1.0000 | 0.9772 | 0.9784 | 0.9773 | 22 | 56.7 |

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
  - Test Accuracy: 1.0000 (100.00%)
  - Test Precision: 1.0000 (100.00%)
  - Test Recall: 1.0000 (100.00%)
  - Best Epoch: 3
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 8.04 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T13:33:44.615411

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
  - Test Precision: 0.9863 (98.63%)
  - Test Recall: 0.9863 (98.63%)
  - Best Epoch: 5
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 11.37 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T13:45:06.979918

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
  - Test Precision: 0.9853 (98.53%)
  - Test Recall: 0.9847 (98.47%)
  - Best Epoch: 5
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 13.54 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T13:58:39.140911

### 5 Subjects

- **Dataset Sizes:**
  - Training: 1421 samples
  - Validation: 246 samples
  - Testing: 241 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9751 (97.51%)
  - Test Precision: 0.9752 (97.52%)
  - Test Recall: 0.9753 (97.53%)
  - Best Epoch: 12
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 23.74 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T14:22:23.355570

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
  - Test Precision: 0.9801 (98.01%)
  - Test Recall: 0.9795 (97.95%)
  - Best Epoch: 8
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 27.98 minutes
- **Best Hyperparameters:**
  - lr: 1.33e-04
  - weight_decay: 5.06e-03
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T14:50:22.438559

### 7 Subjects

- **Dataset Sizes:**
  - Training: 1996 samples
  - Validation: 343 samples
  - Testing: 339 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9587 (95.87%)
  - Test Precision: 0.9613 (96.13%)
  - Test Recall: 0.9591 (95.91%)
  - Best Epoch: 11
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 26.42 minutes
- **Best Hyperparameters:**
  - lr: 6.36e-04
  - weight_decay: 1.77e-04
  - train_batch_size: 32
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T15:16:47.919137

### 8 Subjects

- **Dataset Sizes:**
  - Training: 2288 samples
  - Validation: 391 samples
  - Testing: 388 samples
- **Performance:**
  - Validation Accuracy: 0.9974 (99.74%)
  - Validation Precision: 0.9975 (99.75%)
  - Validation Recall: 0.9974 (99.74%)
  - Test Accuracy: 0.9897 (98.97%)
  - Test Precision: 0.9901 (99.01%)
  - Test Recall: 0.9898 (98.98%)
  - Best Epoch: 23
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 52.22 minutes
- **Best Hyperparameters:**
  - lr: 4.37e-04
  - weight_decay: 1.29e-07
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T16:09:01.091910

### 9 Subjects

- **Dataset Sizes:**
  - Training: 2580 samples
  - Validation: 441 samples
  - Testing: 435 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9632 (96.32%)
  - Test Precision: 0.9651 (96.51%)
  - Test Recall: 0.9632 (96.32%)
  - Best Epoch: 20
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 54.10 minutes
- **Best Hyperparameters:**
  - lr: 4.37e-04
  - weight_decay: 1.29e-07
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T17:03:07.353714

### 10 Subjects

- **Dataset Sizes:**
  - Training: 2871 samples
  - Validation: 491 samples
  - Testing: 483 samples
- **Performance:**
  - Validation Accuracy: 1.0000 (100.00%)
  - Validation Precision: 1.0000 (100.00%)
  - Validation Recall: 1.0000 (100.00%)
  - Test Accuracy: 0.9772 (97.72%)
  - Test Precision: 0.9784 (97.84%)
  - Test Recall: 0.9773 (97.73%)
  - Best Epoch: 22
- **Training Details:**
  - Optuna Trials Completed: 30
  - Training Duration: 56.67 minutes
- **Best Hyperparameters:**
  - lr: 6.70e-04
  - weight_decay: 2.22e-06
  - train_batch_size: 16
  - val_batch_size: 32
- **Timestamp:** 2025-07-31T17:59:47.526450

## Analysis

### Performance Trends

- **Test Performance Range:**
  - Accuracy: 0.9587 - 1.0000
  - Precision: 0.9613 - 1.0000
  - Recall: 0.9591 - 1.0000
- **Validation Performance Range:**
  - Accuracy: 0.9974 - 1.0000
  - Precision: 0.9975 - 1.0000
  - Recall: 0.9974 - 1.0000
- **Average Performance:**
  - Test Accuracy: 0.9793
  - Test Precision: 0.9802
  - Test Recall: 0.9795
  - Validation Accuracy: 0.9997
  - Validation Precision: 0.9997
  - Validation Recall: 0.9997

### Best Performance
- **2 subjects** achieved the highest test accuracy: 1.0000 (100.00%)
  - Test Precision: 1.0000 (100.00%)
  - Test Recall: 1.0000 (100.00%)

### Lowest Performance
- **7 subjects** achieved the lowest test accuracy: 0.9587 (95.87%)
  - Test Precision: 0.9613 (96.13%)
  - Test Recall: 0.9591 (95.91%)

### Notes

1. All experiments used the same random seed for reproducibility
2. Early stopping was applied based on validation accuracy
3. Hyperparameters were optimized using Optuna with TPE sampler
4. Results show performance on the test set using the model with best validation accuracy

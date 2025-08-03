# SVM Scalability Analysis Report

**Generated on:** 2025-08-03 06:24:06

This report analyzes how the SVM model performance scales with an increasing number of subjects (classes).

## Results Summary

| Subjects | Train Samples | Test Samples | Test Acc | Test Prec | Test Rec | Optuna Val Acc | Duration (min) |
|----------|---------------|--------------|----------|-----------|----------|----------------|----------------|
| 2 | 556 | 96 | 0.9896 | 0.9898 | 0.9896 | 1.0000 | 1.2 |
| 3 | 845 | 145 | 0.9931 | 0.9933 | 0.9931 | 1.0000 | 1.8 |
| 4 | 1128 | 193 | 0.9896 | 0.9902 | 0.9896 | 1.0000 | 2.5 |
| 5 | 1421 | 241 | 0.9834 | 0.9844 | 0.9833 | 1.0000 | 3.4 |
| 6 | 1705 | 289 | 0.9343 | 0.9495 | 0.9342 | 0.9966 | 3.4 |
| 7 | 1996 | 339 | 0.9794 | 0.9798 | 0.9795 | 1.0000 | 2.9 |
| 8 | 2288 | 388 | 0.9485 | 0.9518 | 0.9491 | 0.9949 | 4.5 |
| 9 | 2580 | 435 | 0.9195 | 0.9268 | 0.9203 | 0.9909 | 3.0 |
| 10 | 2871 | 483 | 0.9607 | 0.9632 | 0.9611 | 0.9939 | 3.9 |

## Detailed Results & Hyperparameters

### 2 Subjects
- **Test Accuracy:** 0.9896
- **Best Hyperparameters (C):** 1.45e+01
- **Best Hyperparameters (gamma):** 1.88e-03

### 3 Subjects
- **Test Accuracy:** 0.9931
- **Best Hyperparameters (C):** 3.30e+01
- **Best Hyperparameters (gamma):** 1.06e-04

### 4 Subjects
- **Test Accuracy:** 0.9896
- **Best Hyperparameters (C):** 7.76e+01
- **Best Hyperparameters (gamma):** 1.09e-04

### 5 Subjects
- **Test Accuracy:** 0.9834
- **Best Hyperparameters (C):** 2.29e+01
- **Best Hyperparameters (gamma):** 3.55e-04

### 6 Subjects
- **Test Accuracy:** 0.9343
- **Best Hyperparameters (C):** 3.87e-01
- **Best Hyperparameters (gamma):** 7.32e-04

### 7 Subjects
- **Test Accuracy:** 0.9794
- **Best Hyperparameters (C):** 3.30e+01
- **Best Hyperparameters (gamma):** 1.06e-04

### 8 Subjects
- **Test Accuracy:** 0.9485
- **Best Hyperparameters (C):** 4.43e+00
- **Best Hyperparameters (gamma):** 2.53e-04

### 9 Subjects
- **Test Accuracy:** 0.9195
- **Best Hyperparameters (C):** 1.08e+01
- **Best Hyperparameters (gamma):** 7.03e-04

### 10 Subjects
- **Test Accuracy:** 0.9607
- **Best Hyperparameters (C):** 3.63e+00
- **Best Hyperparameters (gamma):** 3.71e-04


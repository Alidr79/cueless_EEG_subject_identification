# SVM Test Results (Seed 42)

Model: SVM with RBF Kernel
Hyperparameters: C=4.120828, gamma=0.001039

## Overall Performance (Seed 42)

- **Accuracy:** 0.9417
- **Macro Precision:** 0.9464
- **Macro Recall:** 0.9416

## Per-Class Accuracy

- **sub-01:** 1.0000
- **sub-02:** 0.9792
- **sub-03:** 0.9592
- **sub-04:** 0.9792
- **sub-05:** 0.8750
- **sub-06:** 0.7292
- **sub-07:** 0.9400
- **sub-08:** 0.9796
- **sub-09:** 0.9574
- **sub-10:** 0.9583
- **sub-11:** 1.0000

## Per-Word Accuracy

- **backward:** 0.9126
- **forward:** 0.9455
- **left:** 0.9417
- **right:** 0.9483
- **stop:** 0.9600

## Confusion Matrix

```
[[48  0  0  0  0  0  0  0  0  0  0]
 [ 0 47  0  0  0  0  0  0  0  0  1]
 [ 1  1 47  0  0  0  0  0  0  0  0]
 [ 0  0  0 47  0  0  0  0  0  0  1]
 [ 1  0  4  0 42  1  0  0  0  0  0]
 [ 0  0  0  0  1 35  0  0 10  0  2]
 [ 0  0  3  0  0  0 47  0  0  0  0]
 [ 0  0  0  0  0  0  0 48  0  0  1]
 [ 2  0  0  0  0  0  0  0 45  0  0]
 [ 0  0  0  0  0  0  0  1  0 46  1]
 [ 0  0  0  0  0  0  0  0  0  0 49]]
```

# 🧠✨ Subject Identification Based On Cueless Imagined Speech
[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2501.09700&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2501.09700)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FFD21E)](https://huggingface.co/datasets/Alidr79/cueless_EEG_subject_identification)

This repository provides tools and scripts for processing EEG data, performing signal preprocessing, and running machine and deep learning models for "**Cueless EEG imagined speech for subject identification: dataset and benchmarks**" [paper](https://arxiv.org/abs/2501.09700).


---

## 📜 Table of Contents
- [🛠 Usage](#-usage)
  - [1. Data Structuring & File Integration](#1-data-structuring--file-integration)
    - 📥 Download the Data
    - 📁 Replace the Directory
    - 🔄 Run the Integration Script
  - [2. Signal Preprocessing](#2-signal-preprocessing)
  - [3. Machine & Deep Learning Models](#3-machine--deep-learning-models)
    - Generating Datasets
    - SVM and XGB on Statistical and Wavelet Features
    - Moment Models
    - End-to-End Architectures
- [📫 Contact](#-contact)

---

## 🛠 Usage
### 1. Data Structuring & File Integration
📥 Download the Data

Download the "Device Output Data" directory from the [Hugging Face Dataset](https://huggingface.co/datasets/Alidr79/cueless_EEG_subject_identification) of this project.

📁 Replace the Directory

Replace the empty `Device Output Data` directory in this repository with the downloaded one:

🔄 Run the Integration Script

Integrate the `.tdms` files with the corresponding `.csv` files for each session to generate the `Dataset` directory containing `.fif` files.
```bash
python data_structuring_file_integration.py --subject-num <SUBJECT_NUM> --session-num <SESSION_NUM>
```
Example:
```bash
python data_structuring_file_integration.py --subject-num 1 --session-num 1
```
⏭️ Alternatively, you can skip this step by downloading the pre-generated `Dataset` directory from the [Hugging Face](https://huggingface.co/datasets/Alidr79/cueless_EEG_subject_identification).

### 2. Signal Preprocessing
Preprocess the EEG signals to prepare them for analysis and modeling.
```bash
python signal_preprocessing.py --subject-num <SUBJECT_NUM> --session-num <SESSION_NUM> --gender <GENDER>
```
Example:
```bash
python signal_preprocessing.py --subject-num 10 --session-num 1 --gender female
```
- Logs: Execution logs are saved in the logs_integrate_structuring directory.
- Output: Preprocessed signals are saved in Dataset/derivatives/preprocessed_eeg.

⏭️ Alternatively, skip this step by downloading the `Dataset/derivatives/preprocessed_eeg` from the [Hugging Face](https://huggingface.co/datasets/Alidr79/cueless_EEG_subject_identification).

### 3. Machine & Deep Learning Models
- **Generating Datasets**

Navigate to the `ml_dl_models` directory to generate `.npy` datasets from the `.fif` files and create train/val/test splits.
```bash
cd ml_dl_models

# Generate numpy datasets from .fif files
# 🚨 Before running the following code change the data_path_template in the code to your desired path
python read_and_save.py

# Generate train and test sets with different configurations
python train_test_generate.py
```
This will generate datasets like `train_dataset.npy` (First 3 sessions of all subjects), `train_dataset_ses-1,2.npy` (First 2 sessions of all subjects), etc which will be used in further steps.
- **SVM and XGB on Statistical and Wavelet Features**

Navigate to the `base_ml_features` directory to replicate results using SVM and XGB with feature extraction.

```bash
cd base_ml_features

# Example: Run Optuna for XGBoost with Wavelet Features
python optuna_XGB_wavelet.py
```
Optuna Studies:

You can visualize hyperparameter tuning results using the Optuna dashboard.

```bash
optuna-dashboard sqlite:///study_XGB_wavelet.db
```
Open the provided URL in your browser to view the tuning plots.

- **Moment Models**

Navigate to the appropriate moment subdirectory based on the desired model size (base, large or small).

Full Fine-Tuning:

Run the following script to full fine-tune the Moment model.

```bash
python full_finetune.py
```
Then for training the SVM or XGB on the output embeddings of the fine-tuned or zero-shot models you could run the following script as an example.
```bash
python optuna_XGB_moment_large.py
```

**Logs:** Final results and logs are available as .md files within the respective directories.

- **End-to-End Architectures**

For end-to-end architectures, navigate to the directories starting with braindecode_. For example, for the EEG Conformer model:
```bash
cd braindecode_EEGConformer

# Run the training script with Optuna hyperparameter tuning
python optuna_EEGConformer.py

```

## 📫 Contact
For any questions or feedback, feel free to reach out:
- Email: ali.derakhshesh79@gmail.com
- linkedin: https://www.linkedin.com/in/ali-derakhshesh/











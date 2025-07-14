import os
import re
import glob
import numpy as np

def parse_results_from_file(file_path):
    """
    Parses a single markdown file to extract test metrics.

    Args:
        file_path (str): The path to the markdown file.

    Returns:
        dict: A dictionary containing 'accuracy', 'precision', and 'recall'
              if found, otherwise None.
    """
    # Regular expression to find the line and capture the float values
    # It looks for "Test Accuracy: ", "Test Precision: ", "Test Recall: "
    # and captures the numbers that follow.
    pattern = re.compile(
        r"Test Accuracy: ([\d.]+),\s*"
        r"Test Precision: ([\d.]+),\s*"
        r"Test Recall: ([\d.]+)"
    )

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    # The match object groups start at 1
                    accuracy = float(match.group(1))
                    precision = float(match.group(2))
                    recall = float(match.group(3))
                    return {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall
                    }
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None
    
    # Return None if the line with metrics is not found in the file
    print(f"Warning: Could not find metrics in {file_path}")
    return None

def main():
    """
    Main function to find result files, parse them, and compute stats.
    """
    # Use glob to find all files matching the pattern in the current directory.
    # The '*' is a wildcard that matches any characters.
    file_pattern = "seed_*_test_results_*.md"
    file_paths = glob.glob(file_pattern)

    if not file_paths:
        print(f"Error: No files found matching the pattern '{file_pattern}'.")
        print("Please make sure the script is in the same directory as your result files.")
        return

    print(f"Found {len(file_paths)} result files to analyze:")
    for path in file_paths:
        print(f"- {os.path.basename(path)}")
    print("-" * 30)

    # Lists to store the collected metrics from all files
    accuracies = []
    precisions = []
    recalls = []

    for path in file_paths:
        results = parse_results_from_file(path)
        if results:
            accuracies.append(results['accuracy'])
            precisions.append(results['precision'])
            recalls.append(results['recall'])

    if not accuracies:
        print("Could not extract any data. Please check the file contents and format.")
        return

    # Convert lists to NumPy arrays for easy statistical calculations
    accuracies_np = np.array(accuracies)
    precisions_np = np.array(precisions)
    recalls_np = np.array(recalls)

    # Calculate mean and standard deviation for each metric
    mean_accuracy = np.mean(accuracies_np)
    std_accuracy = np.std(accuracies_np)

    mean_precision = np.mean(precisions_np)
    std_precision = np.std(precisions_np)

    mean_recall = np.mean(recalls_np)
    std_recall = np.std(recalls_np)

    # Print the final results in a clean format
    print("Statistical Analysis of Test Results:\n")
    print(f"Accuracy:  Mean = {mean_accuracy*100:.4f}, Std Dev = {std_accuracy*100:.4f}")
    print(f"Precision: Mean = {mean_precision*100:.4f}, Std Dev = {std_precision*100:.4f}")
    print(f"Recall:    Mean = {mean_recall*100:.4f}, Std Dev = {std_recall*100:.4f}")


if __name__ == "__main__":
    main()

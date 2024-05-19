import csv
import os
import pandas as pd

result_dir = '../experiments_logs'
output_dir = 'experiment_results'

fed_types = ['FedAvg', 'FedProx', 'SCAFFOLD', 'MOON', 'noFL_multitargets']#, 'noFL_mergedtargets']

def find_subdirectories(directory_path):
    """ Returns a list of subdirectories in the specified directory path. """
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    return subdirectories


# Compile Data
datasets = find_subdirectories(result_dir)
for dataset in datasets:
  if not os.path.exists(os.path.join(output_dir, dataset)):
      os.mkdir(os.path.join(output_dir, dataset))
  TSmethods = find_subdirectories(os.path.join(result_dir, dataset))
  for TSmethod in TSmethods:
    if not os.path.exists(os.path.join(output_dir, dataset, TSmethod)):
      os.mkdir(os.path.join(output_dir, dataset, TSmethod))
      
    # Initialize a DataFrame to store the compiled data
    compiled_data = pd.DataFrame(columns=["fed_type", "acc", "f1_score", "auroc"])

    for fed_type in fed_types:
      data_path = os.path.join(os.path.join(result_dir, dataset, TSmethod), f'results_{fed_type}.csv')
      if os.path.exists(data_path):
          data = pd.read_csv(data_path)
          # Extract mean and std for acc, f1_score, and auroc
          mean_acc = data.loc[data['scenario'] == 'mean', 'acc'].values[0]
          std_acc = data.loc[data['scenario'] == 'std', 'acc'].values[0]
          mean_f1 = data.loc[data['scenario'] == 'mean', 'f1_score'].values[0]
          std_f1 = data.loc[data['scenario'] == 'std', 'f1_score'].values[0]
          mean_auroc = data.loc[data['scenario'] == 'mean', 'auroc'].values[0]
          std_auroc = data.loc[data['scenario'] == 'std', 'auroc'].values[0]
          
          # Combine mean and std into a single value with format "mean ± std"
          acc = f"{mean_acc} +- {std_acc}"
          f1_score = f"{mean_f1} +- {std_f1}"
          auroc = f"{mean_auroc} +- {std_auroc}"
          
          # Append to the compiled DataFrame
          compiled_data = compiled_data.append({
              "fed_type": fed_type,
              "acc": acc,
              "f1_score": f1_score,
              "auroc": auroc
          }, ignore_index=True)
          
          data.to_csv(os.path.join(os.path.join(output_dir, dataset, TSmethod), f"results_{fed_type}.csv"))
      else:
          print(f'{data_path} does not exist!')
      
    # Save the compiled DataFrame to a new CSV file
    compiled_csv_path = os.path.join(os.path.join(output_dir, dataset, TSmethod), f"compiled_results_{dataset}_{TSmethod}.csv")
    compiled_data.to_csv(compiled_csv_path, index=False)


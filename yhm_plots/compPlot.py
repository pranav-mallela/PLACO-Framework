import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

folder_path_baseline = "../output/baseline"
folder_path_yhm = "../output/yhm"

files_baseline = os.listdir(folder_path_baseline)
files_yhm = os.listdir(folder_path_yhm)

files_baseline.sort()
files_yhm.sort()

for i in range(len(files_baseline)):
    #add dimensions to plot
    plt.figure(figsize=(20, 10))

    file_path_baseline = os.path.join(folder_path_baseline, files_baseline[i])
    file_path_yhm = os.path.join(folder_path_yhm, files_yhm[i])
    
    df_baseline = pd.read_csv(file_path_baseline)
    df_yhm = pd.read_csv(file_path_yhm)
    
    baseline_policies = []
    for policy in df_baseline:
        baseline_policies.append(df_baseline[policy].mean())
    
    yhm_policies = []
    for policy in df_yhm:
        yhm_policies.append(df_yhm[policy].mean())

    headers = df_baseline.columns.tolist()

    plt.plot(headers, baseline_policies, color='green', label='baseline')
    plt.plot(headers, yhm_policies, color='red', label='yhm')
    plt.xlabel('policies')
    plt.ylabel('accuracy')
    plt.title(f'{files_baseline[i]}')
    plt.legend()
    plt.xticks(rotation=0)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'./comparison_{files_baseline[i]}.pdf')
    plt.close()
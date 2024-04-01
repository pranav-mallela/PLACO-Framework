import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

num_humans = [5, 7, 10, 15]
test_sizes = ['500', '999', '5000', '9999']
datasets = {'cifar10h':'cnn_data','imagenet':'imagenet_data'}

for dataset,filename in datasets.items():
    for num_human in num_humans:
        avg_test_size_data = []
        min_test_size_data = []
        max_test_size_data = []
        labels = []
        for test_size in test_sizes:
            data = pd.read_csv(f"output/{dataset}/accuracy/{num_human}_{filename}_accuracy_{test_size}.csv")
            labels = data.columns
            # print(labels)
            data = data[1:].to_numpy()
            # print(data)
            avg_test_size_data.append(np.mean(data, axis=0))
            min_test_size_data.append(np.min(data, axis=0))
            max_test_size_data.append(np.max(data, axis=0))
        avg_test_size_data = np.transpose(np.array(avg_test_size_data))
        min_test_size_data = np.transpose(np.array(min_test_size_data))
        max_test_size_data = np.transpose(np.array(max_test_size_data))
        labels= labels[2:]
        avg_test_size_data = avg_test_size_data[2:]
        min_test_size_data = min_test_size_data[2:]
        max_test_size_data = max_test_size_data[2:]

        plt.figure()
        for i in range(len(avg_test_size_data)):
            plt.plot(test_sizes, avg_test_size_data[i], label=labels[i], marker='o', markersize=5)
            plt.fill_between(test_sizes, min_test_size_data[i], max_test_size_data[i], alpha=0.2)
        plt.xlabel('Test Size')
        plt.ylabel('Accuracy')
        # plt.yticks(np.arange(0.5, 1.0, step=0.1))
        plt.title(f'{dataset} {num_human} Humans')
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f'output/plots/accuracy{num_human}_{dataset}.png')
        plt.close()

                


       

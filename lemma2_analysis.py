from combiner import MAPOracleCombiner
from experiment import load_data, simulate_humans
from dependencies import *
import numpy as np

def check_ti_Yhm():
    # Load data
    model_name = 'cnn_data'
    human_counts, model_probs, y_true = load_data(model_name)
    human_counts = human_counts.astype(int)
    y_true = y_true.astype(int)

    # Generate human output from human counts through simulation
    t = simulate_humans(human_counts, y_true, accuracy_list=accuracies)

    # Get confusion matrices
    combiner = MAPOracleCombiner()
    combiner.fit(model_probs, t, y_true)
    hcm_list = combiner.confusion_matrix

    overall_list = []

    for i in range(len(model_probs)):
        post_prob_list = []
        for h in range(len(accuracies)):
            post_prob = 0
            for y in range(len(model_probs[i])):
                post_prob += (model_probs[i][y] * hcm_list[h][t[i][h]][y])
            post_prob_list.append(round(post_prob,2))
        overall_list.append(post_prob_list)
        print(post_prob_list)
    
    human_probs = [0]*len(accuracies)
    for h in range(len(accuracies)):
        for i in range(len(model_probs)):
            if overall_list[i][h] > 0.5:
                human_probs[h] += 1
        human_probs[h] /= len(model_probs)
    
    print(human_probs)
        

check_ti_Yhm()
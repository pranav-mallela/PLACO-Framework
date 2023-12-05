import numpy as np

def calculate_confidence(probability_vector):
    entropy = -np.sum(probability_vector * np.log(probability_vector))
    confidence = 1.0 - entropy
    return confidence

def estimate_human_value(model_probability_vector, human_confusion_matrix, human_accuracy):
    k = human_confusion_matrix.shape[0]

    alignment = np.sum(model_probability_vector * human_confusion_matrix, axis=0)

    # weighted_accuracy = np.sum(alignment) * human_accuracy

    return np.sum(alignment)

def calculate_value_array(model_probability_vector, human_confusion_matrices, human_accuracies):
    num_humans = len(human_confusion_matrices)
    value_array = [0] * num_humans
    model_confidence = calculate_confidence(model_probability_vector)

    for i in range(num_humans):
        human_value = estimate_human_value(model_probability_vector, human_confusion_matrices[i], human_accuracies[i])
        value_array[i] = model_confidence * human_value

    return value_array


# def knapsack_dynamicValue_using_alignment(accuracies, model_prob_list, B, human_confusion_matrices):
#     B = int(B*100)
#     accuracies = [int(a*100) for a in accuracies]
#     instance_wise_budgets = find_instance_wise_budget(model_prob_list, B)
#     subsets = []
#     for i in range(len(instance_wise_budgets)):
#         b = instance_wise_budgets[i]
#         value = calculate_value_array(model_prob_list[i], human_confusion_matrices, accuracies)
#         subsets.append(knapsack(value, b, accuracies))

#     return subsets
def knapsack_dynamicValue_using_alignment(accuracies, model_prob_list, B, human_confusion_matrices):
    B = int(B*100)
    accuracies = [int(a*100) for a in accuracies]
    subsets = []
    values = []
    for i in range(len(model_prob_list)):
        value = calculate_value_array(model_prob_list[i], human_confusion_matrices, accuracies)
        values.append(value)

    B = int(sum(sum(values,[])))
    
    instance_wise_budgets = find_instance_wise_budget(model_prob_list, B)

    for i in range(len(instance_wise_budgets)):
        b = instance_wise_budgets[i]
        subsets.append(knapsack(values[i], b, accuracies))

    return subsets


def find_instance_wise_budget(model_prob_list, B):
    instance_wise_budget = [int((1-max(m))*100) for m in model_prob_list]
    total = sum(instance_wise_budget)
    return [i*B//total for i in instance_wise_budget]

def knapsack(value, capacity, weight):
    chosen_subset = []
    n = len(value)
    dp = [[0 for x in range(capacity + 1)] for x in range(n + 1)]
 
    # Knapsack
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weight[i-1] <= w:
                if value[i-1] + dp[i-1][w-weight[i-1]] > dp[i-1][w]:
                    dp[i][w] = value[i-1] + dp[i-1][w-weight[i-1]]
                else:
                    dp[i][w] = dp[i-1][w]
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtracking to find optimal subset
    i = n
    w = capacity
    while i > 0 and w > 0:
        if dp[i][w] == dp[i-1][w]:
            i -= 1
        else:
            chosen_subset.append(i-1)
            w -= weight[i-1]
            i -= 1
 
    return chosen_subset
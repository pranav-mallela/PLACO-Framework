def knapsack_instance_wise_subsets(accuracies, model_prob_list, B):
    B = int(B*100)
    accuracies = [int(a*100) for a in accuracies]
    return [knapsack(accuracies, b, accuracies) for b in find_instance_wise_budget(model_prob_list, B)]

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
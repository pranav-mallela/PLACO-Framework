from itertools import combinations
import random
import numpy as np
import pandas as pd
from dependencies import accuracies
from valuator import value
from cost import c5 as cost
from test_Yhm import test_Yhm
import pulp
import math

df = None

df_preds = pd.DataFrame()

def log(combiner, hx, tx, mx, num_humans):
    global df

    df = pd.DataFrame(hx)
    cols = []
    for i in range(len(df.columns)):
        cols.append(f'Human {df.columns[i]}')
    df.columns = cols
    df['Model'] = pd.DataFrame(mx)[0]
    df['True'] = pd.DataFrame(tx)[0]
    
    with open(f'./log/Confusion-Matrix.md', 'w') as f:
        f.write("# Confusion Matrix\n")
        for i in range(num_humans):
            f.write(f"## {i}\n")
            for j in combiner.confusion_matrix[i]:
                for k in j:
                    f.write(f"{k: .2f} ")
                f.write("\n")
    return df

def add_predictions(policy_name, predictions):
    df_preds[f'Prediction with {policy_name}'] = predictions

def add_predictions_to_policy():
    for c in df_preds:
        df[c] = df_preds[c]


def update_data_policy(df, optimal, policy_name):
    df[f'Using {policy_name}'] = pd.DataFrame(optimal)[0].transform(lambda x: '['+' '.join(map(str, x))+']')

def dump_policy(i):
    df.to_csv(f"./log/policy_{i}.csv")

def lb_best_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    def f(x):
        return x / (1 - x)

    policy_name = "lb_best_policy"

    optimal = []

    t = 0

    for p in hx:
        m = np.array([[f(combiner.confusion_matrix[i][p[i]][j]) for j in range(num_classes)] for i in range(num_humans)])
        m *= (m > 1)
        m += (m == 0) * 1

        y_opt = tx[t]

        optimal.append([i for i, x in enumerate(m[:, y_opt]) if x != 1])

        t += 1
    
    optimal = np.array(optimal, dtype=object)
    
    return optimal
        
def single_best_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    '''
    return best human only in all the cases
    '''

    policy_name = "single_best_policy"

    # we can estimate accuracy for i_th human as (combiner.confusion_matrix[i].trace() / num_classes)

    accuracy = [(combiner.confusion_matrix[i].trace() / num_classes) for i in range(num_humans)]
    best = accuracy.index(max(accuracy))

    optimal = np.array([None for _ in range(len(hx))], dtype=object)
    for i in range(len(hx)):
        optimal[i] = [best]
    
    return optimal

def mode_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    '''
    return a single human which denotes the mode of the subset
    '''

    policy_name = "mode_policy"

    mode = []

    for t in range(len(hx)):
        majority = [0 for _ in range(num_classes)]
        for i in range(num_humans):
            majority[hx[t][i]] += 1
        mode.append([(list(hx[t])).index(majority.index(max(majority)))])
    
    optimal = np.array([None for _ in range(len(mode))])
    for i in range(len(mode)): optimal[i] = mode[i]

    return mode

def weighted_mode_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    '''
    return a single human which denotes the weighted mode of the subset
    '''

    policy_name = "mode_policy"

    mode = []

    for t in range(len(hx)):
        weighted_majority = [0 for _ in range(num_classes)]
        for i in range(num_humans):
            # weighted_majority[hx[t][i]] += accuracies[i]
            weighted_majority[hx[t][i]] += combiner.confusion_matrix[i].trace() / num_classes
        mode.append([(list(hx[t])).index(weighted_majority.index(max(weighted_majority)))])
    
    optimal = np.array([None for _ in range(len(mode))])
    for i in range(len(mode)): optimal[i] = mode[i]

    return mode

def select_all_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    return np.array([[i for i in range(num_humans)] for _ in range(len(hx))]), [[num_humans]]

def random_policy(combiner, hx, tx, mx, num_humans, num_classes=10):
    '''
    return a random subset
    '''

    random = []
    cost_of_subset = []
    policy_name = "random_policy"

    humans = list(range(num_humans))

    for t in range(len(hx)):
        random_selection = []
        for i in humans:
            if (np.random.random() < 0.5):
                random_selection.append(i)
        
        cost_of_subset.append([(cost(random_selection,mx,combiner.confusion_matrix))])
        random.append(random_selection)

    optimal = np.array(random, dtype=object)

    return optimal,cost_of_subset

def pseudo_lb_best_policy_overloaded(combiner, hx, tx, mx, num_humans, num_classes=10):

    def f(x):
        return x / (1 - x)

    policy_name = "pseudo_lb_best_policy_overloaded"

    optimal = []
    cost_of_subset = []

    for p in hx:
        m = np.array([[f(combiner.confusion_matrix[i][p[i]][j]) for j in range(num_classes)] for i in range(num_humans)])
        m *= (m > 1)
        m += (m == 0) * 1

        y_opt = np.argmax(np.prod(m, axis=0))

        subset = [i for i, x in enumerate(m[:, y_opt]) if x != 1]
        cost_of_subset.append([(cost(subset,mx,combiner.confusion_matrix))])
        optimal.append(subset)
    
    optimal = np.array(optimal, dtype=object)
    
    return optimal,cost_of_subset

def eamc(combiner, hx, tx, mx, num_humans, num_classes=10):
    def g(x, hcm_list, mpv, B):
        if sum(x) == 0:
            return value(x, hcm_list, mpv)
        return value(x, hcm_list, mpv)/(1 - math.exp(-cost([i for i in range(len(x)) if x[i] == 1],mpv,hcm_list)/B))
    
    hcm_list = combiner.confusion_matrix
    mpv_list = mx
    B = num_humans*len(mpv_list[0])*0.375
    humans = []
    cost_of_subset = []
    for mpv in mpv_list:
        u, v, bin = {}, {}, {}
        u[0] = v[0] = [0]*num_humans
        bin[0] = [u[0], v[0]]
        P = [[0]*num_humans]
        for _ in range(50):
            x = random.choice(P)
            x1 = [random.choices([b, b^1], weights=((num_humans-1), 1), k=1)[0] for b in x]
            sz = sum(x1)
            if cost([i for i in range(len(x)) if x1[i] == 1],mpv,hcm_list) <= B:
                if sz not in bin:
                    u[sz] = v[sz] = x1
                    bin[sz] = [u[sz], v[sz]]
                    P.append(x1)
                else:
                    if g(x1, hcm_list, mpv, B) >= g(u[sz], hcm_list, mpv, B):
                        u[sz] = x1
                    if value(x1, hcm_list, mpv) >= value(v[sz], hcm_list, mpv):
                        v[sz] = x1
                    P.remove(bin[sz][0])
                    P.remove(bin[sz][1]) if bin[sz][1] != bin[sz][0] else None
                    P.append(u[sz]); P.append(v[sz])
                    bin[sz][0] = u[sz]
                    bin[sz][1] = v[sz]
            
        x = max(P, key=lambda x: value(x, hcm_list, mpv))
        subset = [i for i in range(len(x)) if x[i] == 1]
        cost_of_subset.append([cost(subset,mx,combiner.confusion_matrix)])
        humans.append(subset)
        print(len(humans)) if len(humans)%1000 == 0 else None

    return humans,cost_of_subset

def pomc(combiner, hx, tx, mx, num_humans, num_classes=10):
    hcm_list = combiner.confusion_matrix
    mpv_list = mx
    humans = []
    cost_of_subset = []
    for mpv in mpv_list:
        P = [[0]*num_humans]
        for _ in range(25):
            x = random.choice(P)
            x1 = [random.choices([b, b^1], weights=((num_humans-1), 1), k=1)[0] for b in x]
            check = False
            f_x1 = value(x1, hcm_list, mpv)
            for z in P:
                f_z = value(z, hcm_list, mpv)
                if (f_x1 > f_z) or (f_x1 >= f_z and cost([i for i in range(len(x)) if x1[i] == 1],mpv,hcm_list) < cost([i for i in range(len(x)) if z[i] == 1],mpv,hcm_list)):
                    check = True
                    break
            if check:
                for z in P:
                    if f_x1 >= f_z and cost([i for i in range(len(x)) if x1[i] == 1],mpv,hcm_list) <= cost([i for i in range(len(x)) if z[i] == 1],mpv,hcm_list):
                        P.remove(z)
                P.append(x1)
        x = max(P, key=lambda x: (value(x, hcm_list, mpv), -cost([i for i in range(len(x)) if x[i] == 1],mpv,hcm_list)))
        subset = [i for i in range(len(x)) if x[i] == 1]
        cost_of_subset.append([cost(subset,mx,combiner.confusion_matrix)])
        humans.append(subset)
    return humans,cost_of_subset


def linear_program(combiner, hx, tx, mx, num_humans, num_classes=10):
    chosen_subsets = []
    costs_of_subsets = []
    hcm_list = combiner.confusion_matrix
    mpv_list = mx
    B = num_humans*len(mpv_list[0])*0.375

    def v_f(hi_list, h, j_star):
        phi = hcm_list[h]
        epsilon = np.max(np.diag(phi))
        p = phi[hi_list[h]][j_star]
        # print(p)
        # return p/(1-p)
        return (p + epsilon) / (2 - (p + epsilon))
    
    def f(x,a):
        # return x/(1-x)
        return (x+a)/(2-(x+a))
    
    for mpv in mpv_list:
        # print("\n\nFor a given instance: ")
        _, est = test_Yhm(hcm_list, mpv)

        # get estimated ground truth: choose the one that maximises the product of all hcm values at estimated human label
        j_star = -1
        maxi = -1
        for j in range(num_classes):
            p = 1
            for i in range(num_humans):
                p *= f(hcm_list[i][est[i]][j], np.max(np.diag(hcm_list[i])))
            if p > maxi:
                maxi = p
                j_star = j
        # print(j_star)

        h_costs = {i : int(np.random.uniform(0.0001, len(mpv_list[0]))) for i in range(num_humans)}
        h_values = {i: v_f(est,i,j_star) for i in range(num_humans)}
        # print(np.argmax(mpv), est)
        # print(h_values)

        # Suppress printing during the solving process
        pulp.LpSolverDefault.msg = 0
    
        # Create a LP problem
        prob = pulp.LpProblem("Optimal Human Subset Selection", pulp.LpMaximize)

        # Define decision variables
        x = pulp.LpVariable.dicts("Select", range(num_humans), 0, 1, pulp.LpBinary)

        # Objective function: maximize the product of the values
        # Note: We take the logarithm of the product to convert the product into a sum
        prob += pulp.lpSum(np.log(h_values[i]) * x[i] for i in range(num_humans))

        # Constraint: total cost must be less than or equal to the budget
        prob += pulp.lpSum(h_costs[i] * x[i] for i in range(num_humans)) <= B

        # Solve the LP problem
        prob.solve()

        # Optimal subset
        optimal_set = [i for i in range(num_humans) if x[i].varValue == 1]
        cost_of_set = sum(h_costs[i] for i in optimal_set)
        chosen_subsets.append(optimal_set)
        costs_of_subsets.append([cost_of_set])

    return chosen_subsets, costs_of_subsets

def check_all(combiner, hx, tx, mx, num_humans, num_classes=10):
    chosen_subsets = []
    costs_of_subsets = []
    hcm_list = combiner.confusion_matrix
    mpv_list = mx
    B = num_humans*len(mpv_list[0])*0.375

    def v_f(hi_list, h, j_star):
        phi = hcm_list[h]
        epsilon = np.max(np.diag(phi))
        p = phi[hi_list[h]][j_star]
        # return p/(1-p)
        return (p + epsilon) / (2 - (p + epsilon))
    
    def f(x,a):
        # return x/(1-x)
        return (x+a)/(2-(x+a))
    
    for m in range(len(mpv_list)):
        # _, est = test_Yhm(hcm_list, mpv)
        est = hx[m]

        # get estimated ground truth: choose the one that maximises the product of all hcm values at estimated human label
        j_star = -1
        maxi = -1
        for j in range(num_classes):
            p = 1
            for i in range(num_humans):
                p *= f(hcm_list[i][est[i]][j], np.max(np.diag(hcm_list[i])))
            if p > maxi:
                maxi = p
                j_star = j

        h_costs = {i : int(np.random.uniform(0.0001, len(mpv_list[0]))) for i in range(num_humans)}
        h_values = {i: v_f(est,i,j_star) for i in range(num_humans)}

        total = 1<<num_humans
        max_value = -1
        best_set = []
        cost_of_best_set = 0
        for i in range(total):
            subset = []
            curr_cost = 0
            curr_value = 1
            for j in range(num_humans):
                if i & (1<<j):
                    curr_cost += h_costs[j]
                    curr_value *= h_values[j]
                    subset.append(j)
            if curr_value > max_value and curr_cost <= B:
                max_value = curr_value
                best_set = subset
                cost_of_best_set = curr_cost
            
        chosen_subsets.append(best_set)
        costs_of_subsets.append([cost_of_best_set])
        # if len(chosen_subsets)%1000 == 0:
            # print(len(chosen_subsets))

    return chosen_subsets, costs_of_subsets


# def knapsack(combiner, hx, tx, mx, num_humans, num_classes=10):
#     chosen_subsets = []
#     costs_of_subsets = []
#     hcm_list = combiner.confusion_matrix
#     mpv_list = mx
#     KS_scaling_factor = 1000
#     B = num_humans*len(mpv_list[0])*0.375*KS_scaling_factor
#     dp = [[0 for _ in range(B + 1)] for _ in range(num_humans + 1)]

#     def v_f(hi_list, h):
#         j_star = -1
#         for j in range(num_classes):
#             p = 1
#             maxi = -1
#             for i in range(num_humans):
#                 p *= hcm_list[i][hi_list[i]][j]
#             if p > maxi:
#                 maxi = p
#                 j_star = j
#         phi = hcm_list[h]
#         epsilon = np.max(np.diag(phi))
#         p = phi[hi_list[h]][j_star]
#         return (p + epsilon) / (2 - (p + epsilon))

#     for mpv in mpv_list:
#         _, est = test_Yhm(hcm_list, mpv)
#         h_costs = {i : int(np.random.uniform(0.0001, len(mpv_list[0]))*KS_scaling_factor) for i in range(num_humans)}
#         h_values = {i: v_f(est,i) for i in range(num_humans)}

#         def c(h):
#             return h_costs[h]

#         def v(h):
#             return h_values[h]
    
#         # Knapsack
#         for i in range(1, num_humans + 1):  
#             for w in range(1, B + 1):
#                 if c(i-1) <= w:
#                     if v(i-1) + dp[i-1][w-c(i-1)] > dp[i-1][w]:
#                         dp[i][w] = v(i-1) * dp[i-1][w-c(i-1)]
#                     else:
#                         dp[i][w] = dp[i-1][w]
#                 else:
#                     dp[i][w] = dp[i-1][w]
        
#         # Backtracking to find optimal subset
#         optimal_set = []
#         cost_of_set = 0
#         i = num_humans
#         w = B
#         while i > 0 and w > 0:
#             if dp[i][w] == dp[i-1][w]:
#                 i -= 1
#             else:
#                 optimal_set.append(i-1)
#                 cost_of_set += c(i-1)
#                 w -= c(i-1)
#                 i -= 1
    
#         chosen_subsets.append(optimal_set)
#         costs_of_subsets.append([cost_of_set])

#     return chosen_subsets, costs_of_subsets

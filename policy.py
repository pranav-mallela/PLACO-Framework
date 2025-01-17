import math
import random
import pulp
import numpy as np
import pandas as pd
from cost_function import c5 as cost
from value_function import value
from estimation_methods import posterior_estimation

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

"""Subset Selection Methods Used in Our Experiments"""
def pseudo_lb_best_policy_overloaded(combiner, hx, tx, mx, num_humans, h_costs, estimated_true_labels, estimated_human_labels, num_classes=10):

    def f(x):
        return x / (1 - x)

    policy_name = "pseudo_lb_best_policy_overloaded"

    optimal = []
    cost_of_subset = []

    for idx, p in enumerate(hx):
        m = np.array([[f(combiner.confusion_matrix[i][p[i]][j]) for j in range(num_classes)] for i in range(num_humans)])
        m *= (m > 1)
        m += (m == 0) * 1

        y_opt = np.argmax(np.prod(m, axis=0))

        subset = [i for i, x in enumerate(m[:, y_opt]) if x != 1]
        cost_of_subset.append([np.sum(h_costs[idx])]) # cost of all humans for instance idx
        optimal.append(subset)
    
    optimal = np.array(optimal, dtype=object)
    
    return optimal,cost_of_subset

def eamc(combiner, hx, tx, mx, num_humans, h_costs, estimated_true_labels, estimated_human_labels, num_classes=10):
    def g(x, hcm_list, mpv, B, idx):
        if sum(x) == 0:
            return value(x, hcm_list, mpv, estimated_true_labels[idx], estimated_human_labels[idx])
        return value(x, hcm_list, mpv, estimated_true_labels[idx], estimated_human_labels[idx])/(1 - math.exp(-cost([i for i in range(len(x)) if x[i] == 1],mpv,hcm_list, h_costs[idx])/B))
    
    hcm_list = combiner.confusion_matrix
    mpv_list = mx
    B = num_humans*len(mpv_list[0])*0.375
    humans = []
    cost_of_subset = []
    for idx, mpv in enumerate(mpv_list):
        u, v, bin = {}, {}, {}
        u[0] = v[0] = [0]*num_humans
        bin[0] = [u[0], v[0]]
        P = [[0]*num_humans]
        for _ in range(50):
            x = random.choice(P)
            x1 = [random.choices([b, b^1], weights=((num_humans-1), 1), k=1)[0] for b in x]
            sz = sum(x1)
            if cost([i for i in range(len(x)) if x1[i] == 1],mpv,hcm_list, h_costs[idx]) <= B:
                if sz not in bin:
                    u[sz] = v[sz] = x1
                    bin[sz] = [u[sz], v[sz]]
                    P.append(x1)
                else:
                    if g(x1, hcm_list, mpv, B, idx) >= g(u[sz], hcm_list, mpv, B, idx):
                        u[sz] = x1
                    if value(x1, hcm_list, mpv, estimated_true_labels[idx], estimated_human_labels[idx]) >= value(v[sz], hcm_list, mpv, estimated_true_labels[idx], estimated_human_labels[idx]):
                        v[sz] = x1
                    P.remove(bin[sz][0])
                    P.remove(bin[sz][1]) if bin[sz][1] != bin[sz][0] else None
                    P.append(u[sz]); P.append(v[sz])
                    bin[sz][0] = u[sz]
                    bin[sz][1] = v[sz]
            
        x = max(P, key=lambda x: value(x, hcm_list, mpv, estimated_true_labels[idx], estimated_human_labels[idx]))
        subset = [i for i in range(len(x)) if x[i] == 1]
        cost_of_subset.append([cost(subset,mpv,combiner.confusion_matrix, h_costs[idx])])
        humans.append(subset)
        print(len(humans)) if len(humans)%1000 == 0 else None

    return humans, cost_of_subset

def brute_force(combiner, hx, tx, mx, num_humans, h_costs, estimated_true_labels, num_classes=10):
    chosen_subsets = []
    costs_of_subsets = []
    hcm_list = combiner.confusion_matrix
    mpv_list = mx
    # B = num_humans*len(mpv_list[0])*0.375

    # def v_f(hi_list, h, j_star):
    #     phi = hcm_list[h]
    #     epsilon = np.max(np.diag(phi))
    #     p = phi[hi_list[h]][j_star]
    #     # return p/(1-p)
    #     # return (p + epsilon) / (2 - (p + epsilon))
    #     return (2 + p - epsilon)/(3 - (p + epsilon))
    
    # for m in range(len(mpv_list)):
    #     est = hx[m]
    for idx, mpv in enumerate(mpv_list):
        # _, est = test_Yhm(hcm_list, mpv)
        # print('Instance', idx)
        # for i in range(num_humans):
        #     p = hcm_list[i][est[i]][est[i]]
        #     # print(i, np.mean(np.diag(hcm_list[i])), p + np.min(np.diag(hcm_list[i])) - (np.max(np.diag(hcm_list[i]))/2 + 1))
        #     # print(i, np.mean(np.diag(hcm_list[i])), p + 2*np.min(np.diag(hcm_list[i])) - 3/2)
        #     print(i, np.mean(np.diag(hcm_list[i])), (p + 2*np.min(np.diag(hcm_list[i])) - 1) / (3 - (p + 2*np.min(np.diag(hcm_list[i])))), p + 2*np.min(np.diag(hcm_list[i])))
        # print("\n\n")

        # h_costs = {i : int(np.random.uniform(0.0001, len(mpv_list[0]))) for i in range(num_humans)}
        # h_values = {i: v_f(est,i,j_star) for i in range(num_humans)}

        total = 1<<num_humans
        max_value = -1
        best_set = []
        cost_of_best_set = 0
        for i in range(total):
            subset = []
            curr_cost = 0
            bit_rep = []
            for j in range(num_humans):
                if i & (1<<j):
                    curr_cost += h_costs[idx][j]
                    # curr_value *= h_values[j]
                    subset.append(j)
                    bit_rep.append(1)
                else:
                    bit_rep.append(0)
            curr_value = value(bit_rep, hcm_list, mpv, estimated_true_labels[idx])
            if curr_value > max_value: # and curr_cost <= B
                max_value = curr_value
                best_set = subset
                cost_of_best_set = curr_cost
            
        chosen_subsets.append(best_set)
        costs_of_subsets.append([cost_of_best_set])

    return chosen_subsets, costs_of_subsets

def PLACO_greedy(combiner, hx, tx, mx, num_humans, h_costs, estimated_true_labels, estimated_human_labels, num_classes=10):
    chosen_subsets = []
    costs_of_subsets = []
    hcm_list = combiner.confusion_matrix
    mpv_list = mx

    for idx, mpv in enumerate(mpv_list):
        subset = []
        subset_cost = 0
        h_max, h_max_val = -1, -1
        for h, hcm in enumerate(hcm_list):
            human = [0]*num_humans
            human[h] = 1
            h_val = value(human, hcm_list, mpv, estimated_true_labels[idx], estimated_human_labels[idx])
            if h_val > h_max_val:
                h_max_val = h_val
                h_max = h
            if h_val > 1:
                subset.append(h)
                subset_cost += h_costs[idx][h]
        if len(subset) == 0:
            subset.append(h_max)
            subset_cost += h_costs[idx][h_max]
        chosen_subsets.append(subset)
        costs_of_subsets.append([subset_cost])
    
    return chosen_subsets, costs_of_subsets

def PLACO_lp(combiner, hx, tx, mx, num_humans, h_costs, estimated_true_labels, estimated_human_labels, num_classes=10):
    chosen_subsets = []
    costs_of_subsets = []
    hcm_list = combiner.confusion_matrix
    mpv_list = mx
    B = num_humans*len(mpv_list[0])*0.05

    for idx, mpv in enumerate(mpv_list):
        h_vals = [value([1 if i == h else 0 for i in range(num_humans)], hcm_list, mpv, estimated_true_labels[idx], estimated_human_labels[idx]) for h in range(num_humans)]

        # Suppress printing during the solving process
        pulp.LpSolverDefault.msg = 0
    
        # Create a LP problem
        prob = pulp.LpProblem("Optimal Human Subset Selection", pulp.LpMaximize)

        # Define decision variables
        x = pulp.LpVariable.dicts("Select", range(num_humans), lowBound=0, upBound=1, cat=pulp.LpBinary)

        # Objective function: maximize the product of the values
        prob += pulp.lpSum(h_vals[i] * x[i] for i in range(num_humans))

        # Constraint: total cost must be less than or equal to the budget
        prob += pulp.lpSum(h_costs[idx][i] * x[i] for i in range(num_humans)) <= B

        # Solve the LP problem
        prob.solve()

        # Optimal subset
        optimal_set = [i for i in range(num_humans) if x[i].varValue == 1]
        cost_of_set = sum(h_costs[idx][i] for i in optimal_set)
        chosen_subsets.append(optimal_set)
        costs_of_subsets.append([cost_of_set])

    return chosen_subsets, costs_of_subsets
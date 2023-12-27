import random
from test_Yhm import test_Yhm
import statistics as st
import math
import numpy as np

# POMC algorithm for subset selection
T = 25
def pomc(n, hcm_list, mpv_list):
    humans = []
    for mpv in mpv_list:
        P = [[0]*n]
        for _ in range(T):
            x = random.choice(P)
            x1 = [random.choices([b, b^1], weights=((n-1), 1), k=1)[0] for b in x]
            check = False
            f_x1 = f(x1, hcm_list, mpv)
            for z in P:
                f_z = f(z, hcm_list, mpv)
                #  and c(x1) <= c(z) -> removed from below line -> does not allow to select subset of greater than existing size
                if (f_x1 > f_z) or (f_x1 >= f_z and c(x1) < c(z)):
                    check = True
                    break
            if check:
                for z in P:
                    if f_x1 >= f_z and c(x1) <= c(z):
                        P.remove(z)
                P.append(x1)
        x = max(P, key=lambda x: (f(x, hcm_list, mpv), -c(x)))
        subset = [i for i in range(len(x)) if x[i] == 1]
        humans.append(subset)
        # print(len(humans)) if len(humans)%1000 == 0 else None
    return humans
    


# value function f
# take class accuracy in the estimated label, divide by variance of the labels in the subset
# support high class acc, penalize high variance
def pf(x, hcm_list, mpv):
    value = 0
    est_subset = []
    label_acc = {}
    _ , est = test_Yhm(hcm_list, mpv)
    for h in range(len(x)):
        if x[h] == 1:
            est_subset.append(est[h])
            value += hcm_list[h][est[h]][est[h]]
            label_acc.setdefault(est[h], []).append(hcm_list[h][est[h]][est[h]])
    if(len(est_subset) > 0):
        likeness = (1-len(set(est_subset))/len(est_subset))
        value *= likeness

    return value

# likeness = (1-len(set(est_subset))/len(est_subset)) * (st.mean([max(values) for values in label_acc.values() if values]))
# likeness = (1-len(set(est_subset))/len(est_subset)) / (1-st.mean([max(values) for values in label_acc.values() if values]))
# likeness = 1 - sum([max(values) for values in label_acc.values() if values])/value
# value *= (len(x) ** 2)
# if len(est_subset) >= 2 and st.variance(est_subset) != 0:
# value /= st.variance(est_subset)
# value *= (1-len(set(est_subset))/len(est_subset)) ** 2
# value *= math.sqrt(1-len(set(est_subset))/len(est_subset))
# value /= (len(set(est_subset))/len(est_subset))

# value function pseudo_lb
def f(x, hcm_list, mpv):

    def func(p):
        return p / (1 - p)

    subset = [i for i in range(len(x)) if x[i] == 1]
    num_classes = 10
    # _, est = test_Yhm(hcm_list, mpv)
    est = np.random.randint(0,10,len(hcm_list))     #random estimation

    m = np.array([[func(hcm_list[i][est[i]][j]) for j in range(num_classes)] for i in subset])
    m *= (m > 1)
    m += (m == 0) * 1

    return np.max(np.prod(m, axis=0))

# cost function c
# constant cost = 1
def c(x):
    return sum(x)
    
import random
from test_Yhm import test_Yhm
import statistics as st
import math
import numpy as np

# EAMC algorithm for subset selection
T = 50
B = 15
def eamc(n, hcm_list, mpv_list):
    humans = []
    for mpv in mpv_list:
        u, v, bin = {}, {}, {}
        u[0] = v[0] = [0]*n
        bin[0] = [u[0], v[0]]
        P = [[0]*n]
        for _ in range(T):
            x = random.choice(P)
            x1 = [random.choices([b, b^1], weights=((n-1), 1), k=1)[0] for b in x]
            sz = sum(x1)
            if c(x1) <= B:
                if sz not in bin:
                    u[sz] = v[sz] = x1
                    bin[sz] = [u[sz], v[sz]]
                    P.append(x1)
                else:
                    if g(x1, hcm_list, mpv) >= g(u[sz], hcm_list, mpv):
                        u[sz] = x1
                    if f(x1, hcm_list, mpv) >= f(v[sz], hcm_list, mpv):
                        v[sz] = x1
                    P.remove(bin[sz][0])
                    P.remove(bin[sz][1]) if bin[sz][1] != bin[sz][0] else None
                    P.append(u[sz]); P.append(v[sz])
                    bin[sz][0] = u[sz]
                    bin[sz][1] = v[sz]
            
        x = max(P, key=lambda x: f(x, hcm_list, mpv))
        subset = [i for i in range(len(x)) if x[i] == 1]
        humans.append(subset)
        print(len(humans)) if len(humans)%1000 == 0 else None
    return humans
            


# value function f
# take class accuracy in the estimated label, divide by variance of the labels in the subset
# support high class acc, penalize high variance
def pf(x, hcm_list, mpv):
    value = 0
    est_subset = []
    _ , est = test_Yhm(hcm_list, mpv)
    for h in range(len(x)):
        if x[h] == 1:
            est_subset.append(est[h])
            value += hcm_list[h][est[h]][est[h]]
    if len(est_subset) > 0:
        likeness = (1-len(set(est_subset))/len(est_subset))
        value *= likeness
    
    return value

# surrogate function g
def g(x, hcm_list, mpv):
    if sum(x) == 0:
        return f(x, hcm_list, mpv)
    return f(x, hcm_list, mpv)/(1 - math.exp(-c(x)/B))

# value function pseudo_lb
def f(x, hcm_list, mpv):

    def func(p):
        return p / (1 - p)

    subset = [i for i in range(len(x)) if x[i] == 1]
    num_classes = 10
    _, est = test_Yhm(hcm_list, mpv)

    m = np.array([[func(hcm_list[i][est[i]][j]) for j in range(num_classes)] for i in subset])
    m *= (m > 1)
    m += (m == 0) * 1

    return np.max(np.prod(m, axis=0))
    

# cost function c
# constant cost = 1
def c(x):
    return sum(x)
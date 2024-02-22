from test_Yhm import test_Yhm
import numpy as np
import random

def c0(x):
    return sum(x)

def c1(x,mpv,hcm_list):
    cost = 0
    for human in x:
        hcm = hcm_list[human]
        classWiseAcc = [hcm[i][i] for i, _ in enumerate(hcm)]
        human_cost = np.sum(classWiseAcc)
        cost+=human_cost
    return cost

def c2(x,mpv,hcm_list):
    cost = 0
    _, el = test_Yhm(hcm_list, mpv)
    for i in range(len(x)):
        if x[i] == 1:
            hcm = hcm_list[i]
            classWiseAcc = hcm[el[i]][el[i]]
            human_cost = classWiseAcc
            cost+=human_cost
    return cost

def c3(x,mpv,hcm_list):
    cost = 0
    for human in x:
        hcm = hcm_list[human]
        classWiseAcc = [hcm[i][i] for i, _ in enumerate(hcm)]
        human_cost = np.dot(classWiseAcc, mpv[human])
        cost+=human_cost
    return cost

def c4(x,mpv,hcm_list):
    cost = 0
    for human in x:
        hcm = hcm_list[human]
        classWiseAcc = [hcm[i][i] for i, _ in enumerate(hcm)]
        weight = [sum(hcm[y][yh] * mpv[y] for y in range(len(mpv))) for yh in range(len(mpv))]
        human_cost = np.dot(classWiseAcc,weight)
        cost+=human_cost
    return cost

def c5(x,mpv,hcm_list):
    cost = np.sum([random.uniform(0.0001, len(mpv)) for _ in range(len(x))])
    return cost
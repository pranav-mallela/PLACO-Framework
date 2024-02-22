from test_Yhm import test_Yhm
import numpy as np

def value(x, hcm_list, mpv):
    def func(p):
        return p / (1 - p)

    subset = [i for i in range(len(x)) if x[i] == 1]
    num_classes = 10
    _, est = test_Yhm(hcm_list, mpv)
    # est = np.random.randint(0,10,len(hcm_list))     #random estimation
    # est = maxmax(hcm_list)                        # max max estimation
    # est = topk(hcm_list,mpv)        # top3 estimation
    m = np.array([[func(hcm_list[i][est[i]][j]) for j in range(num_classes)] for i in subset])
    m *= (m > 1)
    m += (m == 0) * 1

    return np.max(np.prod(m, axis=0))
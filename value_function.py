from estimation_methods import posterior_estimation
import numpy as np
from dependencies import accuracies as acc_list

def value(x, hcm_list, mpv, y_cap, est):
    def func(p,phi,i):
        a = np.min(np.diag(phi))
        if(acc_list[i] >= 0.5):
            if (p + 2*a) < 1:
                return 1e-9
            elif (p + 2*a) >= 2:
                return 1e9
            else:
                return (p + 2*a - 1)/(2 - (p + 2*a))
        else:
            return 1e-9
    
    subset = [i for i in range(len(x)) if x[i] == 1]
    # _, est = posterior_estimation(hcm_list, mpv)
    # est = np.random.randint(0,10,len(hcm_list))     #random estimation
    # est = maxmax(hcm_list)                        # max max estimation
    # est = topk(hcm_list,mpv)        # top3 estimation
    m = np.array([func(hcm_list[i][est[i]][y_cap], hcm_list[i], i) for i in subset])

    return np.prod(m, axis=0)
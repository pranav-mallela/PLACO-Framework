from estimation_methods import posterior_estimation
import numpy as np

def value(x, hcm_list, mpv):
    def func(p,phi):
        epsilon = np.max(np.diag(phi))
        # return (p + epsilon) / (2 - (p + epsilon))
        return (2 + p - epsilon)/(3 - (p + epsilon))

    def old_func(p,phi):
        epsilon = np.max(np.diag(phi))
        return p / (1-p)
    
    subset = [i for i in range(len(x)) if x[i] == 1]
    num_classes = 10
    _, est = posterior_estimation(hcm_list, mpv)
    # est = np.random.randint(0,10,len(hcm_list))     #random estimation
    # est = maxmax(hcm_list)                        # max max estimation
    # est = topk(hcm_list,mpv)        # top3 estimation
    m = np.array([[func(hcm_list[i][est[i]][j],hcm_list[i]) for j in range(num_classes)] for i in subset])

    return np.max(np.prod(m, axis=0))
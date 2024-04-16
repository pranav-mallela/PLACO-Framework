import numpy as np

def posterior_estimation(hcm_list, mpv):
    y_h_star = []
    posterior_star = []
    for hcm in hcm_list:
        posterior_list = []
        for y_h in range(10):
            posterior = 0
            for y in range(10):
                posterior += (hcm[y_h][y] * mpv[y])
            posterior_list.append(posterior)
        posterior_star.append(np.max(posterior_list))
        y_h_star.append(np.argmax(posterior_list))
    return posterior_star, y_h_star

def maxmax(hcm_list):
    estimated_labels = []
    for i in range(len(hcm_list)):
        hcm = hcm_list[i]
        est_label = np.argmax([np.max(hcm[i]) for i in range(len(hcm))])
        estimated_labels.append(est_label)
    return estimated_labels

def topk(hcm_list,mpv):
    top3 = np.argsort(mpv)[-3:]  
    estimated_labels = []
    for i in range(len(hcm_list)):
        pseudo_label = np.random.choice(top3)  
        hcm = hcm_list[i]
        estimated_labels.append(np.argmax(hcm[pseudo_label]))
    return estimated_labels
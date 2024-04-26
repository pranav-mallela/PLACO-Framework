import numpy as np

def posterior_estimation(hcm_list, mpv):
    NUM_CLASSES = len(mpv)
    y_h_star = []
    posterior_star = []
    for hcm in hcm_list:
        posterior_list = []
        for y_h in range(NUM_CLASSES):
            posterior = 0
            for y in range(NUM_CLASSES):
                posterior += (hcm[y_h][y] * mpv[y])
            posterior_list.append(posterior)
        posterior_star.append(np.max(posterior_list))
        y_h_star.append(np.argmax(posterior_list))
    return posterior_star, y_h_star

def maxmax_estimation(hcm_list, mpv):
    estimated_labels = []
    for i in range(len(hcm_list)):
        hcm = hcm_list[i]
        est_label = np.argmax([np.max(hcm[i]) for i in range(len(hcm))])
        estimated_labels.append(est_label)
    return None, estimated_labels

def topk_estimation(hcm_list, mpv):
    top3 = np.argsort(mpv)[-3:]  
    estimated_labels = []
    for i in range(len(hcm_list)):
        pseudo_label = np.random.choice(top3)  
        hcm = hcm_list[i]
        estimated_labels.append(np.argmax(hcm[pseudo_label]))
    return None, estimated_labels

def random_estimation(hcm_list, mpv):
    estimated_labels = [np.random.randint(0,len(hcm)) for hcm in hcm_list]
    return None, estimated_labels
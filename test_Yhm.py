import numpy as np

def test_Yhm(human_conf_list, model_prob_vector):
    cnt = 0
    y_m = np.argmax(model_prob_vector)
    y_h_star = []
    p_term_star = []
    for human_conf in human_conf_list:
        p_term_list = []
        for y_h in range(10):
            p_term = 0
            for y in range(10):
                p_term += (human_conf[y_h][y] * model_prob_vector[y])
            p_term_list.append(p_term)
        p_term_star.append(np.max(p_term_list))
        y_h_star.append(np.argmax(p_term_list))
    for i in range(len(human_conf_list)):
        if(y_h_star[i] == y_m):
            cnt += 1
    return p_term_star, y_h_star

# def maxmax(hcm_list):
#     estimated_labels = []
#     for i in range(len(hcm_list)):
#         hcm = hcm_list[i]
#         est_label = np.argmax([np.max(hcm[i]) for i in range(len(hcm))])
#         estimated_labels.append(est_label)
#     return estimated_labels

# def topk(hcm_list,mpv):
#     top3 = np.argsort(mpv)[-3:]  
#     estimated_labels = []
#     for i in range(len(hcm_list)):
#         pseudo_label = np.random.choice(top3)  
#         hcm = hcm_list[i]
#         estimated_labels.append(np.argmax(hcm[pseudo_label]))
#     return estimated_labels
import numpy as np

def test_Yhm(human_conf_list, model_prob_vector):
    cnt = 0
    y_m = np.argmax(model_prob_vector)
    y_h_star = []
    # added for analysis
    p_term_star = []
    # prior = lambda x: 1/10
    for human_conf in human_conf_list:
        p_term_list = []
        for y_h in range(10):
            p_term = 0
            for y in range(10):
                # p_term += ((human_conf[y_h][y] * prior(y) * model_conf[y_m][y])/p_y_m)
                p_term += (human_conf[y_h][y] * model_prob_vector[y])
            p_term_list.append(p_term)
        # added for analysis
        p_term_star.append(np.max(p_term_list))
        y_h_star.append(np.argmax(p_term_list))
    # added for analysis
    # print("p_term_star: ", p_term_star)
    for i in range(len(human_conf_list)):
        if(y_h_star[i] == y_m):
            cnt += 1
    return p_term_star, y_h_star

def test_Yhm_estimate_human_labels(humans, human_conf_list, model_conf, model_prob_vector_list, human_accuracies):
    labels_instance = []
    bad_humans = []
    for image in range(len(model_prob_vector_list)):
        p_term_star, y_h_star = test_Yhm(humans, human_conf_list, model_prob_vector_list[image])
        # added for analysis
        temp = []
        for i in range(len(p_term_star)):
            if(p_term_star[i]*human_accuracies[i] < 0.25):
                temp.append(i)
        bad_humans.append(temp)
        labels_instance.append(y_h_star)
    
    # return np.array(labels_instance)
    return np.array(labels_instance), bad_humans


def enhance_estimated_human_labels(human_subset_list, estimated_labels, true_human_labels):
    enhanced_estimated_labels = estimated_labels
    for image in range(len(estimated_labels)):
        for human in human_subset_list[image]:
            enhanced_estimated_labels[image][human] = true_human_labels[image][human]
    
    return np.array(enhanced_estimated_labels)

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
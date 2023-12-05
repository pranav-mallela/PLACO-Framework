def split_instances(humans, bad_humans):
    above_threshold = []
    below_threshold = []
    for i in range(10000):
        human_subset = humans[i]
        f=0
        for h in human_subset:
            if h in bad_humans[i]:
                f=1
                below_threshold.append(i)
                break
        if f==0:
            above_threshold.append(i)
    return above_threshold, below_threshold

def get_split_acc(above_threshold, below_threshold, y_true, y_comb):
    above_cnt = 0
    for i in above_threshold:
        if y_comb[i] == y_true[i]:
            above_cnt += 1
    below_cnt = 0
    for i in below_threshold:
        if y_comb[i] == y_true[i]:
            below_cnt += 1
    
    print("above_length", len(above_threshold), "below_length", len(below_threshold))

    print("above:", above_cnt/len(above_threshold), "below:", below_cnt/len(below_threshold))

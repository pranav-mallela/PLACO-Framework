import random
from dependencies import *
from combiner import *
from estimation_methods import *
from value_function import value

def load_data(model_name, dataset):
    """ Loads the predictions (human and model) and true labels.
        Datasets: cnn_data (CIFAR-10H), resnet152_imagenet_data_counts (IMAGENET-16H)
    """
    dirname = PROJECT_ROOT

    data_path = os.path.join(dirname, f'dataset/{dataset}/{model_name}.csv')
    data = np.genfromtxt(data_path, delimiter=',')

    if(model_name == 'cnn_data'):
        true_labels = data[:, 0]
        human_counts = data[:, 1:11]
        model_probs = data[:, 11:]

        true_labels = true_labels.astype(int)

    
    elif(model_name == 'imagenet_data'):
        true_labels = data[:, 164]
        human_counts = data[:, 165:181]
        model_probs = data[:, 148:164]

        true_labels = true_labels.astype(int)

    return human_counts, model_probs, true_labels

def simulate_humans(human_counts, y_true, accuracy_list = accuracies, seed=0):
    rng = np.random.default_rng(seed)
    human_labels = []

    assert len(human_counts) == len(y_true), "Size mismatch"

    i = -1

    for data_point in human_counts:
        i += 1
        labels = []
        for accuracy in accuracy_list:
            if (rng.random() < accuracy):
                labels.append(y_true[i])
            else:
                prob = data_point
                prob[y_true] = 0
                if (np.sum(prob) == 0):
                    prob = np.ones(prob.shape)
                    prob[y_true[i]] = 0
                prob /= np.sum(prob)
                labels.append(rng.choice(range(len(data_point)), p = prob))
                
        human_labels.append(labels)
    
    return np.array(human_labels)

def get_acc(y_pred, y_true):
    """ Computes the accuracy of predictions.
    If y_pred is 2D, it is assumed that it is a matrix of scores (e.g. probabilities) of shape (n_samples, n_classes)
    """
    if y_pred.ndim == 1:
        return np.mean(y_pred == y_true)
    print("Invalid Arguments")

def get_run_data(model_probs_te, combiner):
    NUM_INSTANCES = model_probs_te.shape[0]
    NUM_CLASSES = model_probs_te.shape[1]

    # Estimating human labels for a given run, choose h(x) by posterior estimation
    estimated_human_labels = []
    for mpv in model_probs_te:
        _, est = posterior_estimation(combiner.confusion_matrix, mpv)
        estimated_human_labels.append(est)

    # Estimating the ground truth for a given run, choose y_cap that maximizes the value function
    estimated_true_labels = []
    for idx, mpv in enumerate(model_probs_te):
        v = 1e-9
        y_cap = 0
        for j in range(NUM_CLASSES):
            curr_value = value(np.ones(NUM_HUMANS), combiner.confusion_matrix, mpv, j, estimated_human_labels[idx])
            if curr_value > v:
                v = curr_value
                y_cap = j
        estimated_true_labels.append(y_cap)

    # Cost of a human is random, but remains fixed for a given run
    h_costs_for_run = [[np.random.uniform(0.0001, NUM_CLASSES) for _ in range(NUM_HUMANS)] for _ in range(NUM_INSTANCES)]
    
    return estimated_true_labels, h_costs_for_run, estimated_human_labels

def main():
    # n_runs = 10
    # test_sizes = [0.95, 0.9, 0.75, 0.5]
    n_runs = 10
    test_sizes = [0.95]

    # dataset = 'cifar10h'
    # model_names = ['cnn_data']
    dataset = 'imagenet'
    model_names = ['imagenet_data']
    out_fpath = f'./output/{dataset}/'
    os.makedirs(out_fpath, exist_ok=True)

    for test_size in test_sizes:

        for model_name in tqdm(model_names, desc='Models', leave=True):
            # Specify output files
            output_file_acc = out_fpath + f'{str(len(accuracies))}_{model_name}_accuracy_{int((1-test_size)*10000)}'

            # Load data
            human_counts, model_probs, y_true = load_data(model_name, dataset)

            # Generate human output from human counts through simulation
            y_h = simulate_humans(human_counts, y_true, accuracy_list=accuracies)

            POLICIES = [
                # ('linear_program', linear_program, False),
                # ('pseudo_lb', pseudo_lb_best_policy_overloaded, False),
                ('greedy', greedy_policy, False),
                # ('eamc', eamc, False),
            ]

            acc_data = []
            for i in tqdm(range(n_runs), leave=False, desc='Runs'):
                seed = random.randint(1, 1000)

                # Train/test split
                y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                    y_h, model_probs, y_true, test_size=test_size, random_state=i * seed)

                # Test over entire dataset
                y_h_te = y_h
                model_probs_te = model_probs
                y_true_te = y_true

                # Considering the accuracy of the best human only
                acc_h = get_acc(y_h_te[:, 0], y_true_te)
                acc_m = get_acc(np.argmax(model_probs_te, axis=1), y_true_te)

                _acc_data = [acc_h, acc_m]
                
                add_predictions("True Labels", y_true_te)

                combiner = MAPOracleCombiner()
                combiner.fit(model_probs_tr, y_h_tr, y_true_tr)

                # Get run data
                estimated_true_labels, h_costs_for_run, estimated_human_labels = get_run_data(model_probs_te, combiner)

                for policy_name, policy, use_true_labels in POLICIES:
                    # Call to policy() to return human subsets and costs
                    humans, cost = policy(combiner, y_h_te, y_true_te if use_true_labels else None, model_probs_te, NUM_HUMANS, h_costs_for_run, estimated_true_labels, estimated_human_labels, model_probs_te.shape[1])

                    with open(f'./output/{dataset}/subset/{str(len(accuracies))}_{int((1-test_size)*10000)}_{policy_name}.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(humans)
                    with open(f'./output/{dataset}/subset_cost/{str(len(accuracies))}_{int((1-test_size)*10000)}_{policy_name}.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(cost)

                    y_comb_te = combiner.combine(model_probs_te, y_h_te, humans)

                    acc_comb = get_acc(y_comb_te, y_true_te)
                    _acc_data.append(acc_comb)

                acc_data += [_acc_data]

            header_acc = ['human', 'model'] + [policy_name for policy_name, _, _ in POLICIES]
            with open(f'{output_file_acc}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header_acc)
                writer.writerows(acc_data)

main()
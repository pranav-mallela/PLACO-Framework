from dependencies import *
from combiner import *
import random
from test_Yhm import *
from sklearn import metrics

def load_data(model_name, dataset):
    """ Loads the predictions (human and model) and true labels.
        Datasets: cnn_data (CIFAR-10H), resnet152_imagenet_data_counts (IMAGENET-16H), lstm_hate_speech_data_clean (Hate Speech and Offensive Language)
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
    
    elif(model_name == 'lstm_hate_speech_data_clean'):
        true_labels = data[:, 5]
        human_counts = data[:, 2:5]
        model_probs = data[:, 7:]

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

def main():
    n_runs = 10
    # test_sizes = [0.999, 0.99, 0.9, 0.0001]
    test_sizes = [0.95, 0.9, 0.5, 0.0001] # use this test_sizes for experiments

    # cost_function = 'random'
    dataset = 'imagenet'
    out_fpath = f'./output/{dataset}/'
    os.makedirs(out_fpath, exist_ok=True)
    model_names = ['imagenet_data']

    for test_size in test_sizes:

        for model_name in tqdm(model_names, desc='Models', leave=True):
            # Specify output files
            output_file_acc = out_fpath + f'{str(len(accuracies))}_{model_name}_accuracy_{int((1-test_size)*10000)}'

            # Load data
            human_counts, model_probs, y_true = load_data(model_name, dataset)

            # Generate human output from human counts through simulation
            y_h = simulate_humans(human_counts, y_true, accuracy_list=accuracies)

            POLICIES = [
                # ('single_best_policy', single_best_policy, False),
                # ('mode_policy', mode_policy, False),
                # ('weighted_mode_policy', weighted_mode_policy, False),
                # ('select_all_policy', select_all_policy, False),
                # ('random', random_policy, False),
                # ('lb_best_policy', lb_best_policy, True),
                ('pseudo_lb', pseudo_lb_best_policy_overloaded, False),
                ('pomc', pomc, False),
                ('eamc', eamc, False),
                ('linear_program', linear_program, False),
                ('check_all', check_all, False),
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

                acc_h = get_acc(y_h_te[:, 0], y_true_te) # considering the accuracy of the best human only
                acc_m = get_acc(np.argmax(model_probs_te, axis=1), y_true_te)

                
                _acc_data = [acc_h, acc_m]
                
                add_predictions("True Labels", y_true_te)

                combiner = MAPOracleCombiner()

                combiner.fit(model_probs_tr, y_h_tr, y_true_tr)

                # confusion_matrix_model = metrics.confusion_matrix(y_true_te, np.argmax(model_probs_te, axis=1))

                for policy_name, policy, use_true_labels in POLICIES:

                    # generate estimated human labels
                    # y_h_te, bad_humans = test_Yhm_estimate_human_labels(len(accuracies), combiner.confusion_matrix, confusion_matrix_model, model_probs_te, accuracies)
                    humans,cost = policy(combiner, y_h_te, y_true_te if use_true_labels else None, model_probs_te, NUM_HUMANS, model_probs_te.shape[1])
                    with open(f'./output/{dataset}/subset/{str(len(accuracies))}_{int((1-test_size)*10000)}_{policy_name}.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(humans)
                    with open(f'./output/{dataset}/subset_cost/{str(len(accuracies))}_{int((1-test_size)*10000)}_{policy_name}.csv', 'w', newline='') as f:
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
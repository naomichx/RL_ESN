import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import json
import time

def nextnonexistent(f):
    """ Check if a file already exist, if yes, creates a new one with the next number at the end.
    Example: input.txt exists, input_1.txt created.
    Input:
    f: text file name
    Output: new file name or same file name."""
    fnew = f
    root, ext = os.path.splitext(f)
    i = 0
    while os.path.exists(fnew):
        i += 1
        fnew = '%s_%i%s' % (root, i, ext)
    return fnew

def define_path(deterministic,delay, gen_test_indiv=False, val_gen_test=False,temp_gen_test=False):
    if deterministic:
        path = 'results/deterministic/'
    else:
        path = 'results/stochastic/'
    if gen_test_indiv and val_gen_test:
        path += 'gen_test_indiv/'
    elif val_gen_test and not gen_test_indiv:
        path += 'val_gen_test/'
    elif temp_gen_test:
        path += 'temp_gen_test/'
    else:
        path += 'delay_' + str(delay) + '/'

    return path

def define_path_same_params(deterministic,delay, overlap,gen_test_indiv=False, val_gen_test=False,temp_gen_test=False):
    if deterministic:
        path = 'results/same_parameters/deterministic/'
    else:
        path = 'results/same_parameters/stochastic/'
    if gen_test_indiv and val_gen_test:
        path += 'gen_test_indiv/'
    elif val_gen_test and not gen_test_indiv:
        path += 'val_gen_test/'
    elif temp_gen_test:
        if overlap:
            path +='temp_gen_test_overlap_easier/'
        else:
            path += 'temp_gen_test_no_overlap/'
    else:
        path += 'delay_' + str(delay) + '/'

    return path

def setup_saving_files(model_type, path, task_file,model_file,n_seed,i, testing=False):
    with open(os.path.join(os.path.dirname(__file__), model_file)) as f:
        model_parameters = json.load(f)
    with open(task_file, "r") as task_f:
        task_parameters = json.load(task_f)
    if testing:
        specific_path = path + model_type+ '_testing' + '/'
    else:
        specific_path = path + model_type + '/'
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    # Check whether the specified path exists or not
    isExist = os.path.exists(specific_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(specific_path)
        print("The new directory is created!")
    print('Folder to save:', specific_path)
    index = 0
    while json_exists(specific_path + "task_{}".format(str(index)) + ".json"):
        index += 1
    task_json = specific_path + "task_{}".format(str(index)) + ".json"
    model_json = specific_path + "model_{}".format(str(index)) + ".json"
    with open(task_json, "w") as to:
        json_string = json.dumps(task_parameters)
        json.dump(json_string, to)
    with open(model_json, "w") as to:
        json_string = json.dumps(model_parameters)
        json.dump(json_string, to)
    if n_seed > 1:
        seed_path = specific_path + '/{}/'.format(i)
        isExist = os.path.exists(seed_path)
        if not isExist:
            os.makedirs(seed_path)
        else:
            print("The new directory was already created!")
        specific_path = seed_path
    else:
        specific_path = None
    return specific_path

def save_files(exp,specific_path, testing=False):
    if testing:
        legal_choices_array = exp.legal_choices_array_testing
        success_array = exp.success_array_testing
        success_array_best_first = exp.success_array_best_first_testing
        success_array_best_last = exp.success_array_best_last_testing
        #all_W_out = exp.model.all_W_out_testing
        record_output_activity = exp.model.record_output_activity
        all_trials = exp.all_trials_testing
    else:
        legal_choices_array = exp.legal_choices_array
        success_array = exp.success_array
        success_array_best_first = exp.success_array_best_first
        success_array_best_last = exp.success_array_best_last
        #all_W_out = exp.model.all_W_out
        record_output_activity = exp.model.record_output_activity
        all_trials = exp.all_trials

    np.save(arr=legal_choices_array['session 0'],
            file=nextnonexistent(
                specific_path + 'legal_choice_array_n_train_{}'.format(exp.task.parameters["nb_train"]) + '.npy'))

    np.save(arr=success_array['session 0'],
            file=nextnonexistent(
                specific_path + 'array_n_train_{}'.format(exp.task.parameters["nb_train"]) + '.npy'))

    np.save(arr=success_array_best_first['session 0'],
            file=nextnonexistent(
                specific_path + 'best_first_array_n_train_{}'.format(exp.task.parameters["nb_train"]) + '.npy'))

    np.save(arr=success_array_best_last['session 0'],
            file=nextnonexistent(
                specific_path + 'best_last_array_n_train_{}'.format(exp.task.parameters["nb_train"]) + '.npy'))

    #np.save(arr=all_W_out,
     #       file=nextnonexistent(specific_path + 'all_W_out' + '.npy'))

    np.save(arr=record_output_activity,
            file=nextnonexistent(specific_path + 'output_activity' + '.npy'))

    for name in ('all_trials', 'delay', 'input_time_1','input_time_2'):
        np.save(arr=all_trials['session 0'][name],
                file=nextnonexistent(specific_path + name + '.npy'))


def json_exists(file_name):
    return os.path.exists(file_name)


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)*100
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def draw_trial(trial):
    from matplotlib import colors
    colormap = colors.ListedColormap(["white", "black"])
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(trial, alpha=0.7, cmap=colormap)

    ax.set_xlabel("Positions", size=14)
    ax.set_ylabel("Cues", size=14)
    ax.set_xticks([0, 1, 2, 3], size=14,
                  color="black")
    ax.set_yticks([0, 1, 2, 3], size=14,
                  color="black")
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    plt.show()

def merge_two_dicts(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def plot_several_results(list_of_arrays, labels, avg_filter = 200, title = 'Percentage of success', save=False,
                         filename_to_save=None):
    cmap = plt.get_cmap("tab20b", 20)
    C = [cmap(2 + i * 4) for i in range(len(list_of_arrays))]
    plt.subplots(figsize=(12, 5))
    for i in range(len(list_of_arrays)):
        res = moving_average(list_of_arrays[i], avg_filter)
        plt.plot(res, color=C[i], label=labels[i])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.ylim((15, 100))
    plt.xlabel('Trial number')
    plt.ylabel('% of success with average filter n={}'.format(avg_filter))
    plt.subplots_adjust(right=0.75)
    if save:
        plt.savefig(nextnonexistent(filename_to_save + '.pdf'))
    plt.show()

def plot_success(success_array, save=False, folder_to_save=None, name='succes_rate', show=True, title = 'Percentage of success'):
    """
    Plot the evolution of training of one model. y-axis: success percentage. x-axis: trial number/50.
        parameters:
            success_array: numpy array
                            contains the percentage of success every 50 time steps
            save: boolean
            folder_to_save: str
            name : str
    """
    plt.subplots(figsize=(10, 5))
    plt.plot(success_array, color='black')
    plt.title(title)
    plt.xlabel('Trial number')
    plt.ylabel(title + ' with avg filter')
    plt.ylim((15, 100))
    if save:
        plt.savefig(nextnonexistent(folder_to_save + name + '.pdf'))
    if show:
        plt.show()

def plot_output_activity(output_array, n, save=False, folder_to_save=None, deterministic=False,name='output_activity', show=True,
                         title='Output neuron activity'):

    for i in range(len(output_array)-n, len(output_array)):
        plt.subplots(figsize=(10, 5))
        trial_info = output_array[i]['trial_info']

        #print(output_array[i])
        output_array[i]['output_activity'] = np.transpose(output_array[i]['output_activity'])
        for k in range(4):
            #print(np.shape(output_array[i]['output_activity'][k][:]))
            if trial_info[k] == 0 or trial_info[k] == -0.01:
                plt.plot(output_array[i]['output_activity'][k][:], alpha=0.2)
            else:
                plt.plot(output_array[i]['output_activity'][k][:], alpha=1)
        if deterministic:
            plt.legend(['Output 0, r = {}'.format(trial_info[0]), 'Output 1, r = {}'.format(trial_info[1]),
                        'Output 2, r = {}'.format(trial_info[2]), 'Output 3, r = {}'.format(trial_info[3])],
                       bbox_to_anchor=(1., 0.5))
        else:
            plt.legend(['Output 0, p_r = {}'.format(trial_info[0]), 'Output 1, p_r = {}'.format(trial_info[1]),
                        'Output 2, p_r = {}'.format(trial_info[2]), 'Output 3, p_r = {}'.format(trial_info[3])],
                        bbox_to_anchor=(1., 0.5))

        if trial_info['best_cue_first']:
            plt.title(title + ' when best cue appears first')
        else:
            plt.title(title + ' when best cue appears last')
        plt.xlabel('Time step')
        plt.ylabel(title)
        plt.subplots_adjust(right=0.75)
        if save:
            isExist = os.path.exists(folder_to_save + 'output_activity_fig/')
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(folder_to_save + 'output_activity_fig/')
                print("The new directory is created!")
            plt.savefig(nextnonexistent(folder_to_save + 'output_activity_fig/' + str(i) + '.pdf'))
        if show:
            plt.show()

def plot_w_out(w_out_array, save=False, folder_to_save=None, name='w_out', show=True,
                         title = 'Output weight value'):

    fig, axs = plt.subplots(nrows=4, figsize=(10, 5))
    w_out_array=np.transpose(w_out_array)

    for i in range(4):
        sub_w = np.transpose(w_out_array[i])

        #axs[i].plot(sub_w[1:,:50])
        axs[i].plot(sub_w[:, :50])
    plt.title(title)
    plt.xlabel('Trial number')
    plt.ylabel(title)
    if save:
        plt.savefig(folder_to_save + name + '.pdf')
    if show:
        plt.show()

def plot_gen_test_results(results, save=True, filename='results.pdf'):
    """
    Plot the results of the generalization test with the standard deviation and the mean.
        parameters:
            input: npy file containing a nested dict of shape:
                    {'n_unseen ..': {'percent_success': [.., .., ..], 'mean': .., 'std': ..},
                     'n_unseen ..': {'percent_success': [.., .., ..], 'mean': .., 'std': ..},
                     ...}
    """
    #results = np.load(npy_file, allow_pickle=True).item()
    x = []
    y = []
    y_lower = []
    y_upper = []
    for n_unseen in results.keys():
        x.append(int(n_unseen[-2:]))
        y.append(results[n_unseen]['mean'])
        y_lower.append(results[n_unseen]['mean'] - results[n_unseen]['std'])
        y_upper.append(results[n_unseen]['mean'] + results[n_unseen]['std'])
    plt.plot(x, y, ls='-', color='black')
    plt.fill_between(x, y_lower, y_upper, facecolor='grey', alpha=0.5)
    plt.ylim((0, 100))
    plt.title('Generalization test: successful decisions on unseen combinations')
    plt.xlabel('Number of unseen combinations')
    plt.ylabel('% of successful decisions averaged on 10 trials ')
    if save:
        plt.savefig(nextnonexistent(filename))
    plt.show()




def plot_indiv_gen_test_results(results, save=True, filename='results.pdf'):
    #results = np.load(npy_file, allow_pickle=True).item()
    Y = results['percent_success']
    barwidth = 0.5
    X = barwidth
    Y_mean = results['mean']
    Y_std = results['std']
    fig = plt.figure()
    cmap = plt.get_cmap("tab20b", 20)
    C = [cmap(2 + i * 4) for i in range(len(Y))]
    plt.errorbar(X, Y_mean, xerr=0, yerr=Y_std,
                fmt=".", color="black", capsize=0, capthick=2,
                elinewidth=2)
    plt.bar(X, height=Y_mean, width=0.9 * barwidth,
           color=C, edgecolor="white")

    plt.scatter([X for i in range(len(Y))], Y, s=20, facecolor="black",
               edgecolor="white")
    plt.ylim(0,100)
    plt.ylabel("% on successful decision on unseen trials")
    plt.xticks([])

    fig.suptitle('Generalization test: training on individual cues')
    #plt.tight_layout()
    if save:
        plt.savefig(nextnonexistent(filename + '.pdf'))
    plt.show()



def W_initializer_SW_1000(*shape, seed=42, sr=None, **kwargs):
    G = nx.watts_strogatz_graph(n=500, k=200, p=0.85) #p small: regular graph, p=1 completely random
    W = nx.adjacency_matrix(G)
    W= W.toarray().astype(np.float64)
    #plt.spy(W,markersize=2)
    #plt.show()
    return W




def connect(P, degree=90, k=10):
    """
    Build a connection matrix W
    """
    n = len(P)
    dP = P.reshape(1, n, 2) - P.reshape(n, 1, 2)  # calculate all possible vectors
    # Distances
    D = np.hypot(dP[..., 0], dP[..., 1]) / 1000.0  # calculate the norms of all vectors
    # Non-isotropic connections
    A = np.zeros((n, n))
    for i in range(n):
        A[i] = np.arctan2(dP[i, :, 1], dP[i, :, 0]) * 180.0 / np.pi  # angles in degrees between all
    W = np.zeros((n, n))
    I = np.argsort(D, axis=1) #indexes that sort the array i axis 1
    for i in range(n):
        R = D[i]
        for j in range(1, n):
            if A[i, I[i, j]] > degree or A[i, I[i, j]] < degree: # connected only from behind
            #if A[i, I[i, j]] > 90 or A[i, I[i, j]] < -90:  # connected only from behind
                W[i, I[i, j]] = (np.random.uniform(0, 1) < np.exp(-(R[I[i, j]] ) ** 2 /0.01))
                #W[i, I[i, j]] = 1
    return W

def W_initializer_1000(*shape, seed=42, sr=None, **kwargs):
    " Create initialization function for W"
    filename = 'topology_test/CVT-1000-seed-0.npy'
    cells_pos = np.load(filename)
    W = connect(cells_pos)
    return W




def W_initializer_300(*shape, seed=42, sr=None, **kwargs):
    " Create initialization function for W"
    filename = 'topology_test/uniform-1024x1024-stipple-300.npy'
    cells_pos = np.load(filename)
    W = connect(cells_pos)
    return W

def W_initializer_cluster_scale_free_1000(*shape, seed=42, sr=None, **kwargs):
    all_w = []
    for i in range(4):
        #G = nx.scale_free_graph(250, alpha=0.25, beta=0.5, gamma=0.25, delta_in=0, delta_out=0)
        G = nx.powerlaw_cluster_graph(250, 10, 0.1, seed=None)
        #G = nx.watts_strogatz_graph(250, 10, 0.5, seed=None)
        w = nx.adjacency_matrix(G)
        w = w.toarray()
        all_w.append(w)
    W0 = np.hstack((all_w[0], np.zeros((250, 250)), np.zeros((250, 250)), np.zeros((250, 250))))
    W1 = np.hstack((np.zeros((250, 250)), all_w[1], np.zeros((250, 250)), np.zeros((250, 250))))
    W2 = np.hstack((np.zeros((250, 250)), np.zeros((250, 250)), all_w[2], np.zeros((250, 250))))
    W3 = np.hstack((np.zeros((250, 250)), np.zeros((250, 250)), np.zeros((250, 250)), all_w[3]))
    W = np.vstack((W0, W1, W2, W3))
    ## Fully connected backbones neurons
    for i in range(0, 751, 250):
        for j in range(0, 751, 250):
            if i != j:
                W[i, j] = np.random.uniform(-1, 1)
                W[j, i] = np.random.uniform(-1, 1)
    indexes = np.nonzero(W)
    for i in range(len(indexes[0])):
        row = indexes[0][i]
        col = indexes[1][i]
        W[row, col] = np.random.uniform(-1, 1)
    return W



import numpy as np
from utils import plot_success, plot_output_activity, plot_w_out, moving_average, plot_several_results
import scipy as sc
import matplotlib.pyplot as plt
from scipy import stats

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def plot_means(mean_success, mean_first, mean_last,mean_legal_choice, labels, avg):
    res_success = []
    res_best_first = []
    res_best_last = []
    res_legal_choices = []
    for model in labels:
        res_success.append(mean_success[model])
        res_best_first.append(mean_first[model])
        res_best_last.append(mean_last[model])
        res_legal_choices.append(mean_legal_choice[model])
    plot_several_results(res_success, labels, avg_filter = avg, title = 'Percentage of success', save=False,
                         filename_to_save=None)


model_types = ('regular',  'forward', 'differential', 'dual', 'dual_separate_input', 'parallel') #, 'converging')
model_types = ('regular',   'dual', 'dual_separate_input')
#model_types = ('dual_separate_input', 'dual', 'converging', 'regular', 'converging_separate_input', 'continuous_forward')

deterministic = False
temp_gen_test = True
val_gen_test = False
delay = 20
testing = True
avg_filter = 50
save = True
n_seeds = 11

if deterministic:
    path = "results/different_parameters/deterministic/"
else:
    path = 'results/same_parameters/stochastic/'

if temp_gen_test:
    #path += 'temp_gen_test_overlap_easier/'
    path += 'temp_gen_test_no_overlap/'
elif val_gen_test:
    path += 'val_gen_test/'
else:
    path += 'delay_{}'.format(str(delay)) + '/'


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'right')


def plot_training(model_types,n_seeds, n_train=1000):
    labels = []
    paths = []
    for model_type in model_types:
        paths.append(path + model_type + '/')
        labels.append(model_type)
    res_success = []
    res_best_first = []
    res_best_last = []
    res_legal_choices = []

    mean_success = {}
    mean_first = {}
    mean_last = {}
    mean_legal_choice = {}
    for k in range(len(paths)):
        success_arrays = []
        best_first_arrays = []
        best_last_arrays = []
        legal_choice_arrays = []
        for i in range(1, 11):
            if i == 12:
                pass
            else:
                success_arrays.append(np.load(paths[k] + str(i) + '/array_n_train_'+ str(n_train)+'.npy', allow_pickle=True))
                best_first_arrays.append(np.load(paths[k]  + str(i) + '/best_first_array_n_train_'+ str(n_train)+'.npy',
                                                 allow_pickle=True))
                best_last_arrays.append(np.load(paths[k] +  str(i) + '/best_last_array_n_train_'+ str(n_train)+'.npy',
                                                allow_pickle=True))
                legal_choice_arrays.append(np.load(paths[k] + str(i) + '/legal_choice_array_n_train_'+ str(n_train)+'.npy',
                                                   allow_pickle=True))
        mean_success[k] = np.mean(np.array(success_arrays), axis=0)
        mean_first[k], error = tolerant_mean(np.array(best_first_arrays))
        mean_last[k], error = tolerant_mean(np.array(best_last_arrays))
        mean_legal_choice[k] = np.mean(np.array(legal_choice_arrays), axis=0)
        #np.save(arr=mean_success, file=path + 'mean_success.npy')
        #np.save(arr=mean_first, file=path +'mean_first.npy')
        #np.save(arr=mean_last, file=path + 'mean_last.npy')

    for model in range(len(paths)):
        res_success.append(mean_success[model])
        res_best_first.append(mean_first[model])
        res_best_last.append(mean_last[model])
        res_legal_choices.append(mean_legal_choice[model])

    plot_several_results(res_success, labels, avg_filter=avg_filter,
                         title='Average performance of {} simulations'.format(n_seeds), save=save,
                         filename_to_save=path + 'success')

    plot_several_results(res_best_first, labels, avg_filter=avg_filter,
                         title = 'Average success when best cue appears at first of {} simulations'.format(n_seeds), save=save,
                         filename_to_save=path + 'best_first')

    plot_several_results(res_best_last, labels, avg_filter=avg_filter,
                         title='Average success when best cue appears at last of {} simulations'.format(n_seeds), save=save,
                         filename_to_save=path + 'best_last')

    plot_several_results(res_legal_choices, labels, avg_filter=avg_filter,
                         title='Average amount of legal choice of {} simulations'.format(n_seeds), save=save,
                         filename_to_save=path + 'legal_choice')


def compute_t_test(model_1, model_2, n_seeds, n_train = 1000):
    labels = []
    paths = []
    for model_type in (model_1, model_2):
        paths.append(path + model_type + '_testing' + '/')
        labels.append(model_type)
    success_arrays = {}
    best_first_arrays = {}
    best_last_arrays = {}
    legal_choice_arrays = {}
    for k in range(len(paths)):
        success_arrays[k] = []
        best_first_arrays[k] = []
        best_last_arrays[k] = []
        legal_choice_arrays[k] = []
        for i in range(1,11):
            if i == 12:
                pass
            else:
                success_arrays[k].append(
                    np.mean(np.load(paths[k] + str(i) + '/array_n_train_' + str(n_train) + '.npy', allow_pickle=True)))
                best_first_arrays[k].append(
                    np.mean(np.load(paths[k] + str(i) + '/best_first_array_n_train_' + str(n_train) + '.npy',
                            allow_pickle=True)))
                best_last_arrays[k].append(np.mean(np.load(paths[k] + str(i) + '/best_last_array_n_train_' + str(n_train) + '.npy',
                                                allow_pickle=True)))
                legal_choice_arrays[k].append(np.mean(
                    np.load(paths[k] + str(i) + '/legal_choice_array_n_train_' + str(n_train) + '.npy',
                            allow_pickle=True)))
    t_test = {}
    t_test['success'] = stats.ttest_rel(success_arrays[0], success_arrays[1])
    t_test['best_first'] = stats.ttest_rel(best_first_arrays[0], best_first_arrays[1])
    t_test['best_last'] = stats.ttest_rel(best_last_arrays[0], best_last_arrays[1])
    t_test['legal_choice'] = stats.ttest_rel(legal_choice_arrays[0], legal_choice_arrays[1])
    return t_test



def plot_testing(model_types, n_seeds, n_train = 1000):
    cmap = plt.get_cmap("tab20b", 20)
    labels = []
    paths = []
    for model_type in model_types:
        paths.append(path + model_type + '_testing' + '/')
        labels.append(model_type)

    success = {}
    first = {}
    last = {}
    legal_choice = {}

    success['mean'] = []
    first['mean'] = []
    last['mean'] = []
    legal_choice['mean'] = []

    success['std'] = []
    first['std'] = []
    last['std'] = []
    legal_choice['std'] = []

    for k in range(len(paths)):
        success_arrays = []
        best_first_arrays = []
        best_last_arrays = []
        legal_choice_arrays = []

        for i in range(n_seeds):
            if i == 12:
                pass
            else:
                success_arrays.append(
                    np.mean(np.load(paths[k] + str(i) + '/array_n_train_' + str(n_train) + '.npy', allow_pickle=True)))
                best_first_arrays.append(
                    np.mean(np.load(paths[k] + str(i) + '/best_first_array_n_train_' + str(n_train) + '.npy',
                            allow_pickle=True)))
                best_last_arrays.append(np.mean(np.load(paths[k] + str(i) + '/best_last_array_n_train_' + str(n_train) + '.npy',
                                                allow_pickle=True)))
                legal_choice_arrays.append(np.mean(
                    np.load(paths[k] + str(i) + '/legal_choice_array_n_train_' + str(n_train) + '.npy',
                            allow_pickle=True)))

        success['mean'].append(np.mean(np.array(success_arrays)*100, axis=0))
        first['mean'].append(np.mean(np.array(best_first_arrays)*100, axis=0))
        last['mean'].append(np.mean(np.array(best_last_arrays)*100, axis=0))
        legal_choice['mean'].append(np.mean(np.array(legal_choice_arrays)*100, axis=0))

        success['std'].append(np.std(np.array(success_arrays) * 100, axis=0))
        first['std'].append(np.std(np.array(best_first_arrays) * 100, axis=0))
        last['std'].append(np.std(np.array(best_last_arrays) * 100, axis=0))
        legal_choice['std'] = np.std(np.array(legal_choice_arrays) * 100, axis=0)

    keys = ['success', 'best_first', 'best_last', 'legal_choice']
    end_titles = ['', ' when best cue appears first', ' when best cue appears last', ' for the legal choices']
    for k,list in enumerate((success, first, last, legal_choice)):
        print(list)
        labels_ttest = []
        for model in model_types:
            labels_ttest.append('p val:' + str(round(compute_t_test('regular', model, n_seeds=n_seeds)[keys[k]][1], 4)))

        fig, ax = plt.subplots(figsize=(10, 5))
        C = [cmap(2 + i * 4) for i in range(len(list['mean']))]
        ax.errorbar(labels, list['mean'], yerr=list['std'], color='black', fmt=".", elinewidth=2)
        #ax.bar(labels, height=list['mean'], color=C, edgecolor="white", label=labels_ttest)
        for i in range(len(labels)):
            ax.bar(labels[i], list['mean'][i], color=C[i], label=labels_ttest[i])
        addlabels(labels, [round(num, 1) for num in list['mean']])

        ax.set_ylabel('% of right choice')
        ax.set_title('Performances per model during testing '+ end_titles[k])
        ax.legend(title='Paired t-test with regular model', bbox_to_anchor=(1.05,1.05))
        ax.set_ylim((0, 100))
        plt.tight_layout()
        plt.show()


plot_training(model_types,n_seeds, n_train=1000)
plot_testing(model_types, n_seeds, n_train = 1000)


"""t_test_success, t_test_best_first, t_test_best_last = compute_t_test('regular', 'dual_separate_input', n_seeds, n_train = 1000)
print('t_test_success:', t_test_success)
print('t_test_best_first:', round(t_test_best_first[1],4))
print('t_test_best_last:', t_test_best_last)"""
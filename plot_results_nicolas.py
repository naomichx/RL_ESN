import numpy as np
import matplotlib.pyplot as plt
from utils import tolerant_mean, nextnonexistent
np.random.seed(123)
cmap = plt.get_cmap("tab20b", 20)

def plot_results_nicolas(ax, Y, names, xlabel, legend=True, yaxis=True, save= False):
    """
    """

    barwidth = 1
    X = barwidth * np.arange(len(Y))
    Y_mean = [np.mean(y) for y in Y]

    Y_std = [np.std(y) for y in Y]
    C = [cmap(2 + i * 4) for i in range(len(Y))]

    ax.errorbar(X, Y_mean, xerr=0, yerr=Y_std,
                fmt=".", color="black", capsize=0, capthick=2,
                elinewidth=2)
    ax.bar(X, height=Y_mean, width=0.9 * barwidth,
           color=C, edgecolor="white")

    for i in range(len(X)):
        if legend:
            ax.text(1, 1 - i * 0.04, names[i], ha="right", va="top",
                    transform=ax.transAxes, color=C[i])

        ax.scatter(X[i] + np.random.normal(0, barwidth / 8,
                                           Y[i].size),
                   Y[i], s=20, facecolor="black",
                   edgecolor="white")

    ax.set_ylim(0, 100)
    if yaxis:
        ax.set_ylabel("Performance", labelpad=-20,
                      fontsize="large")
        ax.set_yticks([0, 100])
        ax.spines['left'].set_position(('data', -barwidth / 2 - 0.1))
    else:
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)

    ax.set_xlabel(xlabel, labelpad=5, fontsize="large")
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['bottom'].set_position(('data', -1))




#

n_seeds = 11
n_train = 1000
n_last = 500
save = True
folder_to_save = '/Users/nchaix/Desktop/plot_nicolas_style/'

path = "/Users/nchaix/Desktop/deterministic/"
sub_folders = ['5-5-5-1-11/', '5-5-5-1-15/']#, '5-5-5-1-20/', '5-5-5-1-30/']
xlabel = ['Delay 0', 'Delay 11', 'Delay 15', 'Delay 20', 'Delay 30']

sub_folders = ['5-5-5-1-11/']
sub_fold = ['5-5-5-1-11/']
xlabel = ['Delay 11']

type = 'success' # 'best_first', best_last, legal_choice, success

fig = plt.figure(figsize=(len(sub_folders) * 5, 5))
for s, sub_fold in enumerate(sub_folders):
    #wide_path = path + sub_fold + 'wide/'
    shallow_path = path + sub_fold + 'shallow/'
    deep_path = path + sub_fold + 'deep/'
    deeper_bis_2_path = path + sub_fold + 'deeper_bis_2/'
    deeper_path = path + sub_fold+ 'deeper/'
    deeper_bis_path = path + sub_fold + 'deeper_bis/'


    #wide_close_path = path + sub_fold+'wide_close/'
    #wide_close_fb_path = path + sub_fold + 'wide_close_fb/'
    #dual_structured_path = path + sub_fold + 'dual_structured/'
    #paths = [shallow_path, deep_path, deeper_path, wide_close_path]
    #labels = ['Random model', 'Structured model', 'Deep structured pathway', 'Dual pathway fb']

    #paths = [shallow_path, deep_path, deeper_path, deeper_bis_path, deeper_bis_2_path]
    #labels = ['Random model', 'Structured model','Deep 3 reservoirs','Deep 4 reservoirs', 'Deep 2 reservoirs']

    paths = [shallow_path, deep_path, deeper_bis_2_path, deeper_path, deeper_bis_path]
    labels = ['Random model', 'Structured model', 'Deep 2 reservoirs', 'Deep 3 reservoirs', 'Deep 4 reservoirs']



    Y = []
    mean_first = []
    mean_last = []
    mean_legal_choice = []
    for k in range(len(paths)):
        success_arrays = []
        best_first_arrays = []
        best_last_arrays = []
        legal_choice_arrays = []
        success_temp_gen_test = []
        for i in range(0, n_seeds):
            if i == 3:
                pass
            else:
                success_arrays.append(
                    np.mean(np.load(paths[k] + str(i) + '/array_n_train_' + str(n_train) + '.npy', allow_pickle=True)[-n_last:])*100)
                best_first_arrays.append(np.mean(np.load(paths[k] + str(i) + '/best_first_array_n_train_' + str(n_train) + '.npy',
                                                 allow_pickle=True)[-n_last:])*100)
                best_last_arrays.append(np.mean(np.load(paths[k] + str(i) + '/best_last_array_n_train_' + str(n_train) + '.npy',
                                                allow_pickle=True)[-n_last:])*100)
                legal_choice_arrays.append(
                    np.mean(np.load(paths[k] + str(i) + '/legal_choice_array_n_train_' + str(n_train) + '.npy',
                            allow_pickle=True)[-n_last:])*100)
        if type == 'success':
            title = 'Success rate on the last ' + str(n_last) + ' trials'
            Y.append(np.array(success_arrays))
        elif type == 'best_first':
            title = 'Success rate when best cue appears first on the last ' + str(n_last) + ' trials'
            Y.append(np.array(best_first_arrays))
        elif type == 'best_last':
            title = 'Success rate when best cue appears last on the last ' + str(n_last) + ' trials'
            Y.append(np.array(best_last_arrays))
        elif type == 'legal_choice':
            title = 'Legal on the last ' + str(n_last) + ' trials'
            Y.append(np.array(legal_choice_arrays))

        mean_first.append(np.array(best_first_arrays))
        mean_last.append(np.array(best_last_arrays))
        mean_legal_choice.append(np.array(legal_choice_arrays))


    ax = plt.subplot(1, len(sub_folders), s+1)

    plot_results_nicolas(ax, Y, labels, xlabel[s], legend=True, yaxis=True)


fig.suptitle(title)
plt.tight_layout()
if save:
    plt.savefig(nextnonexistent(folder_to_save +type +'_n_last_' + str(n_last) + '.pdf'))

plt.show()

import numpy as np
from utils import plot_success, plot_output_activity, plot_w_out, moving_average, plot_several_results


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)
def create_means():
    path = "/Users/nchaix/Desktop/all_seeds/"
    wide_path = path + 'wide_model/'
    deep_path = path + 'deep_model/'
    shallow_path = path + 'shallow_model/'

    n_units = 500
    lr = 0.3
    output_connectivity = 0.5

    sub_path = 'n_units_{}'.format(n_units) + '/lr_{}'.format(lr) + '/output_co_{}'.format(output_connectivity) + '/'

    mean_success = {}
    mean_first = {}
    mean_last = {}
    mean_legal_choice = {}
    succes_arrays = []
    best_first_arrays = []
    best_last_arrays = []
    legal_choice_arrays = []
    models = ['wide', 'shallow', 'deep']
    k = 0
    for path in (wide_path, shallow_path, deep_path):
        for i in (0,1,4,5,9):
            succes_arrays.append(np.load(path + sub_path + str(i) + '/array_n_train_13000.npy', allow_pickle=True))
            best_first_arrays.append(np.load(path + sub_path + str(i) + '/best_first_array_n_train_13000.npy',
                                             allow_pickle=True))
            best_last_arrays.append(np.load(path + sub_path + str(i) + '/best_last_array_n_train_13000.npy',
                                             allow_pickle=True))
            legal_choice_arrays.append(np.load(path + sub_path + str(i) + '/legal_choice_array_n_train_13000.npy',
                                             allow_pickle=True))
        mean_success[models[k]] = np.mean(np.array(succes_arrays), axis=0)
        #mean_first[models[k]] = np.mean(np.array(best_first_arrays), axis=0)
        mean_first[models[k]], error = tolerant_mean(np.array(best_first_arrays))
        #mean_last[models[k]] = np.mean(np.array(best_last_arrays), axis=0)
        mean_last[models[k]], error = tolerant_mean(np.array(best_last_arrays))
        mean_legal_choice[models[k]] = np.mean(np.array(legal_choice_arrays), axis=0)
        np.save(arr=mean_success[models[k]], file=  path + sub_path + 'mean_success.npy')
        #np.save(arr=mean_first[models[k]], file=path + sub_path + 'mean_first.npy')
        #np.save(arr=mean_last[models[k]], file=path + sub_path + 'mean_last.npy')
        k += 1
    return mean_success, mean_first, mean_last, mean_legal_choice
def plot_means(mean_success, mean_first, mean_last,mean_legal_choice, labels = ['wide', 'shallow', 'deep'], avg=200):
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


model_types = ('regular',  'forward', 'parallel', 'converging', 'dual', 'differential')

deterministic = True
delay = 20

if deterministic:
    path = "results/deterministic/"
else:
    path = 'results/stochastic/'

path += 'delay_{}'.format(str(delay)) + '/'

labels = []
paths = []
for model_type in model_types:
    paths.append(path + model_type + '/')
    labels.append(model_type)



res_success = []
res_best_first = []
res_best_last = []
res_legal_choices = []
avg_filter = 50
save = True
n_seeds = 11


mean_success = {}
mean_first = {}
mean_last = {}
mean_legal_choice = {}

n_train = 1000
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
            #success_arrays.append(np.load(paths[k] + sub_path[k] + str(i) + '/array_n_train_13000.npy', allow_pickle=True))
            success_arrays.append(np.load(paths[k] + str(i) + '/array_n_train_'+str(n_train)+'.npy', allow_pickle=True))
           # best_first_arrays.append(np.load(paths[k] + sub_path[k] + str(i) + '/best_first_array_n_train_13000.npy',
            #                                 allow_pickle=True))
            best_first_arrays.append(np.load(paths[k]  + str(i) + '/best_first_array_n_train_'+str(n_train)+'.npy',
                                             allow_pickle=True))
            #best_last_arrays.append(np.load(paths[k] + sub_path[k] + str(i) + '/best_last_array_n_train_13000.npy',
             #                                allow_pickle=True))
            best_last_arrays.append(np.load(paths[k] +  str(i) + '/best_last_array_n_train_'+str(n_train)+'.npy',
                                            allow_pickle=True))
            #legal_choice_arrays.append(np.load(paths[k] + sub_path[k] + str(i) + '/legal_choice_array_n_train_13000.npy',
             #                                allow_pickle=True))
            legal_choice_arrays.append(np.load(paths[k] + str(i) + '/legal_choice_array_n_train_'+str(n_train)+'.npy',
                                               allow_pickle=True))

    mean_success[k] = np.mean(np.array(success_arrays), axis=0)
    #mean_first[k] = np.mean(np.array(best_first_arrays), axis=0)
    #mean_first[k] = [np.mean(a) for a in np.array(best_first_arrays)]
    mean_first[k], error = tolerant_mean(np.array(best_first_arrays))
    #mean_last[k] = np.mean(np.array(best_last_arrays), axis=0)
    #mean_first[k] = [np.mean(a) for a in np.array(best_last_arrays)]
    mean_last[k], error = tolerant_mean(np.array(best_last_arrays))
    mean_legal_choice[k] = np.mean(np.array(legal_choice_arrays), axis=0)
    np.save(arr=mean_success, file=path + 'mean_success.npy')
    np.save(arr=mean_first, file=path +'mean_first.npy')
    np.save(arr=mean_last, file=path + 'mean_last.npy')


for model in range(len(paths)):
    res_success.append(mean_success[model])
    res_best_first.append(mean_first[model])
    res_best_last.append(mean_last[model])
    res_legal_choices.append(mean_legal_choice[model])


n_seeds= 10
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

#mean_success, mean_first, mean_last, mean_legal_choice = create_means()
#plot_means(mean_success, mean_first, mean_last,mean_legal_choice, labels = ['wide', 'shallow', 'deep'], avg=200)

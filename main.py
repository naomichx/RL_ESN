import warnings
import numpy as np
from experiment import Experiment
from utils import plot_success, moving_average, plot_output_activity, plot_w_out, plot_gen_test_results, \
    nextnonexistent, json_exists, plot_indiv_gen_test_results, setup_saving_files, save_files, define_path
import os
import json
import time
import random
warnings.simplefilter(action='ignore', category=FutureWarning)

save = True
model_type = 'forward'  # "forward', 'parallel', 'converging', 'dual', 'differential', 'regular'
deterministic = True
delay = 20    # 11, 30

temp_gen_test = False
val_gen_test = False
save_gen_test = False
gen_test_indiv = False
n_unseen = [5, 10, 20, 30, 40, 50, 60, 70]

n_seed = 11
show_plots = False
avg_filtering = 50

path = define_path(deterministic, delay, gen_test_indiv, val_gen_test, temp_gen_test)


file_gen_test = path + model_type + '/' + model_type + '_gen_test_result.npy'

if __name__ == '__main__':

    if deterministic:
        model_file = 'json_files/deterministic/' + model_type
        task_file = 'json_files/deterministic/task'
    else:
        model_file = 'json_files/stochastic/' + model_type
        task_file = 'json_files/deterministic/task'
    model_file += '_delay_' + str(delay) + '.json'
    task_file += '_delay_' + str(delay) + '.json'
    print('Model: ', model_type, '  Model json file: ', model_file)
    print('Task json file: ', task_file)
    if temp_gen_test:
        print('Proceed to temporal generalization test ...')
    if val_gen_test:
        print('Proceed to value generalization test ...')

    for i in range(n_seed):

        exp = Experiment(model_file=model_file, task_file=task_file, model_type=model_type, deterministic=deterministic,
                        val_gen_test=val_gen_test, gen_test_indiv=gen_test_indiv,
                         temp_gen_test=temp_gen_test, n_unseen=n_unseen, seed=i,
                         save_gen_test=save_gen_test, file_gen_test=file_gen_test)
        if save:
            specific_path = setup_saving_files(model_type, path, task_file, model_file)
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

        exp.run()
        res = moving_average(exp.success_array['session 0'], avg_filtering)
        res_best_first = moving_average(exp.success_array_best_first['session 0'], avg_filtering)
        res_best_last = moving_average(exp.success_array_best_last['session 0'], avg_filtering)
        res_legal_choices = moving_average(exp.legal_choices_array['session 0'], avg_filtering)

        plot_success(res, save=save, folder_to_save=specific_path,
                     name='success_rate_avg_{}'.format(avg_filtering) + 'n_run_{}'.format((str(exp.task.parameters["nb_train"]))),
                     show=show_plots)
        plot_success(res_best_first, save=save, folder_to_save=specific_path,
                     name='best_first_success_rate_avg_{}'.format(avg_filtering) + 'n_run_{}'.format((str(exp.task.parameters["nb_train"]))),
                     show=show_plots, title="Percentage of success when best cue appear at first")

        plot_success(res_best_last, save=save, folder_to_save=specific_path,
                     name='best_last_success_rate_avg_{}'.format(avg_filtering) + 'n_run_{}'.format((str(exp.task.parameters["nb_train"]))),
                     show=show_plots, title="Percentage of success when best cue appear at last")

        plot_success(res_legal_choices, save=save, folder_to_save=specific_path,
                     name='legal_choice_avg_{}'.format(avg_filtering) + 'n_run_{}'.format((str(exp.task.parameters["nb_train"]))),
                     show=show_plots, title="Percentage of legal choices")

        plot_output_activity(exp.model.record_output_activity, n=5, save=save, folder_to_save=specific_path,
                             deterministic=deterministic, show=show_plots)
        #plot_w_out(exp.model.all_W_out, save=True, folder_to_save=specific_path, name='w_out', show=True,
         #          title='Output weight value')

        if save:
            print('Saving all npy files in ', specific_path, '...')
            save_files(exp, specific_path)

        if val_gen_test:
            print('File gen test', file_gen_test)
            if gen_test_indiv:
                np.save(arr=exp.success_indiv_gen_test,
                        file=file_gen_test)
                plot_indiv_gen_test_results(exp.success_indiv_gen_test, save=save_gen_test, filename=file_gen_test[:-3])
            else:
                np.save(arr=exp.success_gen_test,
                        file=file_gen_test)
                plot_gen_test_results(exp.success_gen_test, save=save_gen_test, filename=file_gen_test[:-3])







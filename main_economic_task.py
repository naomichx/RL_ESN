import warnings
import numpy as np
from experiment_economic_task import Experiment
from prospect_fit import *
from utils import plot_success, moving_average, plot_output_activity, plot_w_out, plot_gen_test_results, \
    nextnonexistent, json_exists, plot_indiv_gen_test_results, setup_saving_files, save_files, define_path, define_path_same_params
import os
import json
import time
import random
warnings.simplefilter(action='ignore', category=FutureWarning)

save = False
model_type = 'regular'  # "forward', 'parallel', 'converging', 'dual', 'differential', 'regular'
lottery_type = 'gain' # 'loss', 'all'

n_seed = 1
show_plots = True
avg_filtering = 25

#path = define_path(deterministic, delay, gen_test_indiv, val_gen_test, temp_gen_test)

path = 'results/evo_prospect/'

if __name__ == '__main__':
    task_file = 'json_files/evo_prospect/economic_task.json'
    model_file = 'json_files/evo_prospect/' + model_type  + '.json'#'_PT_' + lottery_type + '.json'
    print('Model: ', model_type, '  Model json file: ', model_file)
    print('Task json file: ', task_file)

    for i in range(0, 1):
        exp = Experiment(seed=i, model_file=model_file, task_file=task_file, model_type=model_type, lottery_type=lottery_type)
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
        plot_success(res, save=save, folder_to_save=specific_path,
                     name='success_rate_avg_{}'.format(avg_filtering) + 'n_run_{}'.format((str(exp.task.parameters["nb_train"]))),
                     show=show_plots)
        exp.plot_control_sigmoid(n_test=500, cond='same_p', type=lottery_type)
        exp.plot_control_sigmoid(n_test=500, cond='same_x', type=lottery_type)
        exp.plot_tradeoff_sigmoid(n_test=500, type=lottery_type)
        df = exp.collect_tradeoff_responses(n_trials=15000, type= lottery_type)
        run_prospect_fit(df)

        if save:
            print('Saving all npy files in ', specific_path, '...')
            save_files(exp, specific_path)










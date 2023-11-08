"""

Hyperparameter optimization of the ESN-RL model using the Optuna and Mlflow libraries.

"""
import os
import json
import numpy as np
import optuna
from typing import Dict
import mlflow
from experiment import Experiment
from utils import moving_average
import random
SEED = 10
random.seed(SEED)
np.random.seed(SEED)

def save_to_disk(path, agent_id, hparams, nrmse):
    try:
        # Create target Directory
        os.mkdir(path + '/' + str(agent_id) + '/')
        print("Directory ", path + '/' + str(agent_id) + '/', " Created ")
    except FileExistsError:
        print("Directory ", path + '/' + str(agent_id) + '/', " already exists")
    with open(path + '/' + str(agent_id) + '/' + 'hparams.json', 'w') as f:
        json.dump(hparams, f)
    np.save(path + '/' + str(agent_id) + '/' + 'nrmse.npy', nrmse)

def get_agent_id(agend_dir_1):
    try:
        os.mkdir(agend_dir_1)
        print("Directory ", agend_dir_1, " Created ")
    except FileExistsError:
        print("Directory ", agend_dir_1, " already exists")
    ids = []
    for id in os.listdir(agend_dir_1):
        try:
            ids.append(int(id))
        except:
            pass
    if ids == []:
        agent_id = 1
    else:
        agent_id = max(ids) + 1
    return str(agent_id)

def sample_hyper_parameters_array(trial: optuna.trial.Trial, n_reservoir) -> Dict:
    sr = {}
    lr = {}
    rc_connectivity = {}
    input_connectivity = {}
    fb_connectivity = {}
    n_units = {}
    for i in range(n_reservoir):
        #sr[i] = trial.suggest_loguniform("sr_{}".format(str(i)), 1e-2, 2)
        sr[i] = trial.suggest_loguniform("sr_{}".format(str(i)), 1e-2, 1)
        lr[i] = trial.suggest_loguniform("lr_{}".format(str(i)), 1e-4, 1)
        #lr[i] = trial.suggest_loguniform("lr_{}".format(str(i)), 1e-4, 0.5)
        input_connectivity[i] = trial.suggest_loguniform("input_connectivity_{}".format(str(i)),  0.1, 1)
        #trial.suggest_float(input_connectivity_{}".format(str(i)),  0.1, 1)
        rc_connectivity[i] = trial.suggest_loguniform("rc_connectivity_{}".format(str(i)), 1e-4, 0.1)
        #fb_connectivity[i] = trial.suggest_loguniform("fb_connectivity_{}".format(str(i)), 1e-4, 1)
        fb_connectivity[i] = trial.suggest_loguniform("fb_connectivity_{}".format(str(i)), 1e-4, 0.2)
    #fb_connectivity[0] = 0
    #fb_connectivity[1] = trial.suggest_loguniform("fb_connectivity_{}".format(str(1)), 1e-4, 0.9)
    output_connectivity = trial.suggest_loguniform("output_connectivity", 0.1, 0.99)
    if n_reservoir == 1:
        n_units = 500
    elif n_reservoir == 2:
        n_units = [250, 250]
    elif n_reservoir == 3:
        n_units = [167, 167, 167]
    print(n_units)

    #n_units = [100, 200, 200]
    #n_units = [30, 270, 200]


    dict = {}
    #beta = trial.suggest_int("beta", 7, 20)
    beta = trial.suggest_int("beta", 7, 15)
    eta = trial.suggest_loguniform("eta", 1e-4, 1e-1)
    #eta = trial.suggest_loguniform("eta", 1e-2, 6e-2)
    decay = trial.suggest_loguniform("decay",  0.3, 0.7)
    #decay = trial.suggest_loguniform("decay", 0.3, 0.5)


    dict['sr'] = sr
    dict['lr'] = lr
    dict['input_connectivity'] =  input_connectivity
    dict['output_connectivity'] = output_connectivity
    dict['rc_connectivity'] = rc_connectivity
    dict['beta'] = beta
    dict['eta']  = eta
    dict['decay'] = decay
    dict['n_units'] = n_units
    dict['fb_connectivity'] = fb_connectivity
    return dict


def objective(trial: optuna.trial.Trial, agent_dir, model_file, task_file, deterministic, model_type, n_res, overlap,
              temp_gen_test=False, val_gen_test=False, n_unseen=None):
    with mlflow.start_run():
        agent_id = get_agent_id(agent_dir)
        mlflow.log_param('agent_id', agent_id)
        arg = sample_hyper_parameters_array(trial, n_res)
        mlflow.log_params(trial.params)
        exp = Experiment(seed=SEED, model_file=model_file, task_file=task_file, model_type=model_type,
                         deterministic=deterministic, val_gen_test=val_gen_test, n_unseen=n_unseen,
                         temp_gen_test=temp_gen_test,
                         hyperparam_optim=True, n_units=arg['n_units'],
                         lr=arg['lr'], sr=arg['sr'],
                         rc_connectivity=arg['rc_connectivity'], fb_connectivity= arg['fb_connectivity'],
                         output_connectivity=arg['output_connectivity'],
                         input_connectivity=arg['input_connectivity'], eta=arg['eta'], beta=arg['beta'],
                         decay=arg['decay'], i_sim=agent_id, overlap= overlap)
        exp.run()
        session_scores = []
        avg = 50
        for i in range(exp.task.parameters["n_session"]):
            session_scores = moving_average(exp.success_array['session 0'], avg)
            session_scores_best_first = moving_average(exp.success_array_best_first['session 0'], avg)
            session_scores_best_last = moving_average(exp.success_array_best_last['session 0'], avg)
            session_scores_legal_choices = moving_average(exp.legal_choices_array['session 0'], avg)

        if val_gen_test:
            print('val gen test')
            score = exp.success_gen_test['n_unseen {}'.format(n_unseen[0])]['mean']
        else:
            score = np.mean(session_scores[-50:])
            #score = np.mean(session_scores_best_first[-50:])
        print('score', score)
        save_to_disk(agent_dir, agent_id, arg, score)
        mlflow.log_metric('percent_success', score)
        return score

def optuna_optim(title, model_type, n_res, overlap, deterministic=False, temp_gen_test=False,
                 val_gen_test=False,  n_unseen=None,  n_trials=600):

    if deterministic:
        model_file = 'json_files/deterministic/' + model_type + '_delay_' + str(delay) + '.json'
        model_file = 'json_files/deterministic/' + model_type + '_temp_gen_test_input_30.json'
        task_file = 'json_files/deterministic/' + 'task_delay_' + str(delay) + '.json'
    else:
        #model_file = 'json_files/stochastic/' + model_type + '_delay_' + str(delay) + '.json'
        model_file = 'json_files/stochastic/' + model_type + '_temp_gen_test.json'
        task_file = 'json_files/stochastic/' + 'task_delay_' + str(delay) + '.json'
    print('task file:', task_file)
    print('model file:', model_file)

    print('Start Optuna optimization ...')
    parent_dir = 'optuna_results'
    title = title
    EXPERIMENT_NAME = 'hyperparameter_search_' + title
    SAVED_AGENTS_DIR = parent_dir + '/mlagent/' + title
    MLFLOW_RUNS_DIR = parent_dir + '/mlflows/' + title

    mlflow.set_tracking_uri(MLFLOW_RUNS_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)

    task_file_path = parent_dir + "/task_params/" + title
    isExist = os.path.exists(task_file_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(task_file_path)
        print("The new directory is created!")

    model_file_path = parent_dir + "/model_params/" + title
    isExist = os.path.exists(model_file_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(model_file_path)
        print("The new directory is created!")
    with open(task_file, "r") as init:
        with open(task_file_path+"/task.json", "w") as to:
            json_string = json.dumps(json.load(init))
            json.dump(json_string, to)
    with open(model_file, "r") as init:
        with open(model_file_path+"/model.json", "w") as to:
            json_string = json.dumps(json.load(init))
            json.dump(json_string, to)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), study_name='optim_' + title,
                                direction='maximize',
                                load_if_exists=True,
                                storage='sqlite:////Users/nchaix/Documents/PhD/code/RL_RC/optuna_results/optuna_db/'
                                        + title + '.db')
    func = lambda trial: objective(trial, agent_dir=SAVED_AGENTS_DIR, model_file=model_file, task_file=task_file,
                                   deterministic=deterministic, n_res=n_res, temp_gen_test=temp_gen_test,
                                   model_type=model_type, val_gen_test=val_gen_test, n_unseen=n_unseen,overlap= overlap )
    study.optimize(func, n_trials=n_trials)
    best_trial = study.best_trial
    hparams = {k: best_trial.params[k] for k in best_trial.params if k != 'seed'}
    print(hparams)


if __name__ == '__main__':
    n_unseen = None

    temp_gen_test = True
    overlap = True
    val_gen_test = False
    n_unseen = [30]

    deterministic = False
    delay = 11
    model_type = "regular"
    n_res = 1

    if temp_gen_test:
        if deterministic:
            title = str(model_type) + '_det_temp_trial_length_limited'
        else:
            if overlap:
                title = str(model_type) + '_stoch_temp_gen_test_overlap_easier'
            else:
                title = str(model_type) + '_stoch_temp_gen_test_no_overlap'
    elif val_gen_test:
        if deterministic:
            title = str(model_type) + '_det_val_gen_test'
        else:
            title = str(model_type) + '_stoch_val_gen_test'
    else:
        if deterministic:
            title = str(model_type) +'_det_delay_{}'.format(str(delay))
        else:
            title = str(model_type) + '_stoch_delay_{}'.format(str(delay))

    optuna_optim(title=title, model_type=model_type, n_res=n_res,
                 deterministic=deterministic, temp_gen_test=temp_gen_test, val_gen_test=val_gen_test, n_unseen=n_unseen,
                 n_trials=600, overlap=overlap)

import json


agent_id = 341
deterministic = False
model_type = "regular"
n_res = 1
delay = 20
temp_gen_test = True
overlap = True
val_gen_test = False

optuna_best_params = '/Users/nchaix/Documents/PhD/code/RL_RC/optuna_results/mlagent/'
if deterministic:
    if temp_gen_test:
        model_file = 'json_files/deterministic/' + model_type + '_temp_gen_test.json'
        optuna_best_params += model_type + '_det_temp_gen_test/' + '{}/'.format(str(agent_id))
    elif val_gen_test:
        model_file = 'json_files/deterministic/' + model_type + '_val_gen_test.json'
        optuna_best_params += model_type + '_det_val_gen_test/' + '{}/'.format(str(agent_id))

    else:
        model_file = 'json_files/deterministic/' + model_type + '_delay_' + str(delay) + '.json'
        optuna_best_params += model_type + '_det_delay_{}'.format(str(delay)) + '/{}/'.format(str(agent_id))

else:
    if temp_gen_test:
        if overlap:
            model_file = 'json_files/stochastic/' + model_type + '_temp_gen_test_overlap_easier.json'
            optuna_best_params += model_type + '_stoch_temp_gen_test_overlap_easier/' + '/{}/'.format(str(agent_id))
        else:
            model_file = 'json_files/stochastic/' + model_type + '_temp_gen_test_no_overlap.json'
            optuna_best_params += model_type + '_stoch_temp_gen_test_no_overlap/' + '/{}/'.format(str(agent_id))


    elif val_gen_test:
        model_file = 'json_files/deterministic/' + model_type + '_val_gen_test.json'
        optuna_best_params += model_type + '_det_val_gen_test/' + '/{}/'.format(str(agent_id))
    else:
        model_file = 'json_files/stochastic/' + model_type + '_delay_' + str(delay) + '.json'
        optuna_best_params += model_type + '_stoch_delay_{}'.format(str(delay)) + '/{}/'.format(str(agent_id))

#optuna_best_params = '/Users/nchaix/Documents/PhD/code/RL_RC/optuna_results/mlagent/'
#optuna_best_params += model_type+ '_stoch_temp_gen_test_no_overlap/'+ '{}/'.format(str(agent_id))

with open(optuna_best_params + "hparams.json", "r") as from_file:
    params = json.load(from_file)
    print(params)

with open(model_file, "r") as to_file:
    model_dict = json.load(to_file)
    if model_type == 'regular':
        for param in ('sr', 'lr', 'input_connectivity', 'rc_connectivity', 'fb_connectivity'):
            model_dict['ESN'][param] = params[param]['0']
        model_dict['output_connectivity'] = params['output_connectivity']
    else:
        for i in range(1, n_res+1):
            for param in ('sr', 'lr', 'input_connectivity', 'rc_connectivity', 'fb_connectivity'):
                model_dict['ESN_{}'.format(str(i))][param] = params[param][str(i-1)]
        model_dict['Readout']['output_connectivity'] = params['output_connectivity']
    for param in ('eta', 'beta', 'decay'):
        model_dict['RL'][param] = params[param]

with open(model_file, "w") as to_file:
    json.dump(model_dict, to_file)

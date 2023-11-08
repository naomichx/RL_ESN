import random
import numpy as np
from economic_task import EconomicTask
from model_economic_task import *
from utils import moving_average, nextnonexistent
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date



class Experiment:
    """
    This class run the whole experiment.

    """
    def __init__(self, seed, model_file, task_file, model_type, lottery_type='all', hyperparam_optim=False,
                 n_units=None, lr=None, sr=None,
                 rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay=None, train_meropi=False):

        random.seed(seed)
        np.random.seed(seed)
        self.model_file = model_file
        self.task_file = task_file
        self.task = EconomicTask(filename=task_file)

        self.model_type = model_type
        self.lottery_type = lottery_type

        self.n_input_time_1 = self.task.n_input_time_1
        self.n_input_time_2 = self.task.n_input_time_2
        self.n_sessions = self.task.n_session
        self.nb_train = self.task.nb_train
        self.seeds = random.sample(range(1, 100), self.n_sessions)
        self.result_sessions = {}
        self.success_array = {}

        for i in range(self.n_sessions):
            self.result_sessions['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.success_array['session {}'.format(i)] = []

        self.count_record = 0
        self.hyperparam_optim = hyperparam_optim
        self.lr = lr
        self.sr = sr
        self.rc_connectivity = rc_connectivity
        self.input_connectivity = input_connectivity
        self.eta = eta
        self.beta = beta
        self.decay = decay
        self.n_units =n_units

        self.train_meropi = train_meropi

    def init_model(self, seed, model_type='shallow'):
        if model_type == "regular":
            print('Initialise Regular model..')
            self.model = Regular(filename=self.model_file, seed=seed,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                 rc_connectivity=self.rc_connectivity, input_connectivity=self.input_connectivity,
                                 eta=self.eta, beta=self.beta, decay=self.decay )
        """elif model_type == "converging":
            print('Initialise Converging model..')
            self.model = Converging(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                               hyperparam_optim=self.hyperparam_optim,  lr=self.lr, sr=self.sr,
                                rc_connectivity=self.rc_connectivity,
                               input_connectivity=self.input_connectivity,
                                eta=self.eta, beta=self.beta,decay=self.decay)
        elif model_type == "converging_func":
            print('Initialise Converging functional model..')
            self.model = Converging_functional(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                               n_units=self.n_units, hyperparam_optim=self.hyperparam_optim,
                                               lr=self.lr, sr=self.sr,rc_connectivity=self.rc_connectivity,
                                               input_connectivity=self.input_connectivity,
                                               eta=self.eta, beta=self.beta,decay=self.decay)
        elif model_type == 'differential':
            print('Initialise Differential  model..')
            self.model = Differential(filename=self.model_file, seed=seed, n_position=self.task.n_pos,n_units=self.n_units,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta,decay=self.decay)
        elif model_type == 'dual':
            print('Initialise Dual  model..')
            self.model = Dual(filename=self.model_file, seed=seed, n_position=self.task.n_pos,n_units=self.n_units,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay)

        elif model_type == 'forward':
            print('Initialise Forward model..')
            self.model = Forward(filename=self.model_file, seed=seed, n_position=self.task.n_pos,n_units=self.n_units,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay)

        elif model_type == 'parallel':
            print('Initialise Parallel model..')
            self.model = Parallel(filename=self.model_file, seed=seed, n_position=self.task.n_pos,n_units=self.n_units,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay)"""

    def count_success(self, model, best_choice, k):
        """
            Store the results of the final choice of the model: 1 if it made the right choice, 0 otherwise.
            parameters:
                task: class object
                model: class object
                trial: array of shape (n_cue, n_positions)
                       current trial of the task.
                    """

        if model.choice == best_choice:
            self.success_array['session {}'.format(k)].append(1)
        else:
            self.success_array['session {}'.format(k)].append(0)


    def get_all_trials(self):
        if self.lottery_type == 'all':
            trials = {}
            trials['all_trials'] = np.concatenate((self.task.control_gain_lotteries['all_trials'],
                                                   self.task.control_loss_lotteries['all_trials']))
            trials['best_choices'] = np.concatenate((self.task.control_gain_lotteries['best_choices'],
                                                     self.task.control_loss_lotteries['best_choices']))
        elif self.lottery_type == 'gain':
            trials = {}
            trials['all_trials'] = self.task.control_gain_lotteries['all_trials']
            trials['best_choices'] = self.task.control_gain_lotteries['best_choices']
        elif self.lottery_type == 'loss':
            trials = {}
            trials['all_trials'] = self.task.control_loss_lotteries['all_trials']
            trials['best_choices'] = self.task.control_loss_lotteries['best_choices']
        return trials


    def process_one_trial(self, task, trial, record_output=True):
        self.count_record += 1
        self.model.record_output_activity[self.count_record] = {}
        self.model.record_output_activity[self.count_record]['output_activity'] = []
        self.model.record_output_activity[self.count_record]['trial_info'] = {}
        trial_chronogram = task.get_trial_with_chronogram(trial)
        self.model.process(trial_chronogram, count_record=self.count_record, record_output=record_output)

    def plot_control_sigmoid(self, n_test, cond, type):
        diff_EV = {}
        if cond == 'all': #same_p, same_x
            trials = self.get_all_trials()
        elif type == 'gain':
            trials = self.task.get_performance_assessment_trials()[0][cond]['all_trials']
        elif type == 'loss':
            trials = self.task.get_performance_assessment_trials()[1][cond]['all_trials']
        for i in range(n_test):
            if int(i % len(trials)) == 0:
                shuffle_list = list(trials)
                random.shuffle(shuffle_list)
                trials = np.array(shuffle_list)
            trial = trials[int(i % len(trials))]
            self.process_one_trial(self.task, trial, record_output=True)
            ev0 = trial[0][0] * trial[0][1]
            ev1 = trial[1][0] * trial[1][1]
            if ev1 - ev0 in diff_EV:
                diff_EV[ev1 - ev0].append(self.model.choice)
            else:
                diff_EV[ev1 - ev0] = [self.model.choice]
        x = []
        y = []

        for key in diff_EV:
            x.append(key)
            y.append(sum(diff_EV[key]) / len(diff_EV[key]))
        plt.xlabel('EV_0 - EV_1')
        plt.ylabel('P(choose 0)')
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        plt.plot(x_sorted, y_sorted, linewidth=0.5, )
        plt.scatter(x, y)
        plt.title(str(type)+ ' '+ str(cond))
        plt.show()

    def collect_tradeoff_responses(self,n_trials, type):
        """
        left: 0, right:1

        """
        if type == 'gain':
            trials = self.task.tradeoff_gain_lotteries['all_trials']

        elif type == 'loss':
            trials = self.task.tradeoff_gain_lotteries['all_trials']
        else:
            trials = np.concatenate((self.task.control_gain_lotteries['all_trials'],
                            self.task.control_loss_lotteries['all_trials']))
        results = {}
        for key in ('P_left', 'P_right', 'V_left', 'V_right', 'response', 'reward', 'gain','subject_id', 'date',
                    'task_id'):
            results[key] = []
        for i in range(n_trials):
            if int(i % len(trials)) == 0:
                random.shuffle(trials)
            trial = trials[int(i % len(trials))]
            self.process_one_trial(self.task, trial, record_output=True)
            results['P_left'].append(trial[0][1])
            results['P_right'].append(trial[1][1])
            results['V_left'].append(trial[0][0])
            results['V_right'].append(trial[1][0])
            results['response'].append(self.model.choice)
            reward = self.task.get_reward(trial, self.model.choice)
            if reward == 0:
                results['reward'].append(0)
                results['gain'].append(0)
            else:
                results['reward'].append(1)
                results['gain'].append(trial[self.model.choice][0])
        results['task_id'] = 6
        results['subject_id'] = 'esn'
        results['date'] = date.today()
        df = pd.DataFrame.from_dict(results)
        #df.to_csv('/Users/nchaix/Documents/PhD/code/RL_RC/results/economic_task/gain/results_sim.csv')
        return df

    def plot_tradeoff_sigmoid(self, n_test, type='gain'):
        diff_EV = {}
        trials = self.task.tradeoff_gain_lotteries['all_trials']
        riskiest_choices = self.task.tradeoff_gain_lotteries['riskiest_choices']
        trial_indexes = [i for i in range(len(trials))]
        for i in range(n_test):
            if int(i % len(trials)) == 0:
                both = list(zip(trials, trial_indexes))
                random.shuffle(both)
                trials, trial_indexes = zip(*both)
                trials, trial_indexes = np.array(trials), np.array(trial_indexes)
            trial = trials[int(i % len(trials))]
            self.process_one_trial(self.task, trial, record_output=True)
            risk_index = riskiest_choices[trial_indexes[int(i % len(trials))]]
            safe_index = 1-risk_index
            assert trial[risk_index][1] < trial[1-risk_index][1]
            ev_riskiest = trial[risk_index][0] * trial[risk_index][1]
            ev_safest = trial[safe_index][0] * trial[safe_index][1]
            if ev_riskiest - ev_safest in diff_EV:
                if self.model.choice == risk_index:
                    diff_EV[ev_riskiest - ev_safest].append(1)
                else:
                    diff_EV[ev_riskiest - ev_safest].append(0)
            else:
                if self.model.choice == risk_index:
                    diff_EV[ev_riskiest - ev_safest] = [1]
                else:
                    diff_EV[ev_riskiest - ev_safest] = [0]
        x = []
        y = []
        for key in diff_EV:
            x.append(key)
            y.append(sum(diff_EV[key]) / len(diff_EV[key]))
        x_sorted, y_sorted = zip(*sorted(zip(x, y)))
        plt.xlabel('EV_riskiest - EV_safest')
        plt.ylabel('P(choose riskiest)')
        plt.scatter(x, y)
        plt.plot(x_sorted,y_sorted,linewidth=0.5,)
        plt.title('Quantity prob. tradeoff ({})'.format(type))
        plt.show()

    def run(self):
        """
        Run the experiment n_sessions times. It is either a generalization test or a normal run.
        At each session, the trial list is first shuffled, before the model executes the task for each trial.
        """
        for k in range(self.n_sessions):
            self.init_model(seed=self.seeds[k], model_type=self.model_type)
            self.count_record = 0
            trials = self.get_all_trials()
            trial_indexes = [i for i in range(len(trials['all_trials']))]
            for i in range(self.nb_train):
                if int(i%len(trials['all_trials'])) == 0:
                    #if i > 1:
                     #   break
                    shuffle_list = list(zip(trials['all_trials'], trials['best_choices'], trial_indexes))
                    random.shuffle(shuffle_list)
                    trials['all_trials'], trials['best_choices'], trial_indexes = zip(*shuffle_list)
                    trials['all_trials'], trials['best_choices'], trial_indexes = np.array(trials['all_trials']),\
                                                                                np.array(trials['best_choices']),\
                                                                                np.array(trial_indexes)

                trial = trials['all_trials'][int(i % len(trials['all_trials']))]
                best_choice = trials['best_choices'][int(i % len(trials['best_choices']))]
                #print('trial', trial)
                #print('best choice', best_choice)
                self.process_one_trial(self.task, trial, record_output=True)
                #print('model choice:', self.model.choice)
                reward, proba = self.task.get_reward(trial, self.model.choice)
                #print(reward, proba)
                if self.train_meropi:
                    trial_index = trial_indexes[int(i % len(trials['all_trials']))]
                    self.model.train_meropi(reward, proba, self.model.choice, trial_index)
                else:
                    self.model.train(reward, proba, self.model.choice)
                self.count_success(self.model, best_choice, k)
            self.result_sessions['session {}'.format(k)] = moving_average(self.success_array['session {}'.format(k)], 50)





























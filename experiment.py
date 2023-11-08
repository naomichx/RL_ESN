import random
import numpy as np
from task import Task
from model import *
from utils import moving_average, nextnonexistent


class Experiment:
    """
    This class run the whole experiment.

    """
    def __init__(self, seed, model_file, task_file, model_type, deterministic,
                 val_gen_test=False, temp_gen_test=False, gen_test_indiv=False, n_unseen=None,
                 hyperparam_optim=False, n_units=None, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 fb_connectivity=None, eta=None, beta=None, save_gen_test=False, output_connectivity=None,
                 file_gen_test=None, decay=None, testing=None, i_sim=0, overlap=True):

        random.seed(seed)
        np.random.seed(seed)
        self.model_file = model_file
        self.task_file = task_file
        self.task = Task(filename=task_file)
        self.n_trial = len(self.task.trials)
        self.model_type = model_type
        self.deterministic = deterministic
        print("Deterministic:", deterministic)
        self.n_input_time_1 = self.task.n_input_time_1
        self.n_input_time_2 = self.task.n_input_time_2
        self.n_sessions = self.task.n_session
        self.nb_train = self.task.nb_train
        self.seeds = random.sample(range(1, 100), self.n_sessions)

        self.init_count()
        self.trial_counter = 0
        self.success_counter = 0
        self.count_record = 0
        self.val_gen_test = val_gen_test
        self.temp_gen_test = temp_gen_test
        self.gen_test_indiv = gen_test_indiv
        self.testing = testing
        self.n_unseen = n_unseen

        if self.val_gen_test:
            if self.gen_test_indiv:
                self.success_indiv_gen_test = {}
                self.success_indiv_gen_test['percent_success'] = []
                self.success_indiv_gen_test['mean'] = None
                self.success_indiv_gen_test['std'] = None
            else:
                self.success_gen_test = {}
                for n in self.n_unseen:
                    self.success_gen_test['n_unseen {}'.format(n)] = {}
                    self.success_gen_test['n_unseen {}'.format(n)]['percent_success'] = []
                    self.success_gen_test['n_unseen {}'.format(n)]['mean'] = None
                    self.success_gen_test['n_unseen {}'.format(n)]['std'] = None
        self.save_gen_test = save_gen_test
        self.file_gen_test = file_gen_test
        self.hyperparam_optim = hyperparam_optim
        self.lr = lr
        self.sr = sr
        self.rc_connectivity = rc_connectivity
        self.output_connectivity = output_connectivity
        self.input_connectivity = input_connectivity
        self.fb_connectivity = fb_connectivity
        self.eta = eta
        self.beta = beta
        self.decay = decay
        self.n_units =n_units

        self.i_sim = i_sim
        self.overlap = overlap


    def init_count(self):
        self.result_sessions = {}
        self.success_array = {}
        self.success_array_best_first = {}
        self.success_array_best_last = {}
        self.legal_choices_array = {}
        self.all_trials = {}
        for i in range(self.n_sessions):
            self.result_sessions['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.success_array['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.success_array_best_first['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.success_array_best_last['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.legal_choices_array['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.all_trials['session {}'.format(i)] = {}
            self.all_trials['session {}'.format(i)]['all_trials'] = []
            self.all_trials['session {}'.format(i)]['delay'] = []
            self.all_trials['session {}'.format(i)]['input_time_1'] = []
            self.all_trials['session {}'.format(i)]['input_time_2'] = []


    def init_count_testing(self):
        self.result_sessions_testing = {}
        self.success_array_testing = {}
        self.success_array_best_first_testing = {}
        self.success_array_best_last_testing = {}
        self.legal_choices_array_testing = {}
        self.all_trials_testing = {}
        for i in range(self.n_sessions):
            self.result_sessions_testing['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.success_array_testing['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.success_array_best_first_testing['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.success_array_best_last_testing['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.legal_choices_array_testing['session {}'.format(i)] = []
        for i in range(self.n_sessions):
            self.all_trials_testing['session {}'.format(i)] = {}
            self.all_trials_testing['session {}'.format(i)]['all_trials'] = []
            self.all_trials_testing['session {}'.format(i)]['delay'] = []
            self.all_trials_testing['session {}'.format(i)]['input_time_1'] = []
            self.all_trials_testing['session {}'.format(i)]['input_time_2'] = []

    def init_model(self, seed, model_type='shallow'):
        if model_type == "regular":
            print('Initialise Regular model..')
            self.model = Regular(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                 rc_connectivity=self.rc_connectivity, input_connectivity=self.input_connectivity,
                                 eta=self.eta, beta=self.beta, decay=self.decay,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity)
        elif model_type == "converging":
            print('Initialise Converging model..')
            self.model = Converging(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                               hyperparam_optim=self.hyperparam_optim,  lr=self.lr, sr=self.sr,
                                rc_connectivity=self.rc_connectivity,
                               input_connectivity=self.input_connectivity,
                                eta=self.eta, beta=self.beta,decay=self.decay,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity)
        elif model_type == "converging_separate_input":
            print('Initialise Converging separate input model..')
            self.model = Converging_separate_input(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                               hyperparam_optim=self.hyperparam_optim,  lr=self.lr, sr=self.sr,
                                rc_connectivity=self.rc_connectivity,
                               input_connectivity=self.input_connectivity,
                                eta=self.eta, beta=self.beta, decay=self.decay, i_sim =self.i_sim)
        elif model_type == "converging_func":
            print('Initialise Converging functional model..')
            self.model = Converging_functional(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                               n_units=self.n_units, hyperparam_optim=self.hyperparam_optim,
                                               lr=self.lr, sr=self.sr, rc_connectivity=self.rc_connectivity,
                                               input_connectivity=self.input_connectivity,
                                               eta=self.eta, beta=self.beta,decay=self.decay)
        elif model_type == 'differential':
            print('Initialise Differential  model..')
            self.model = Differential(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity)
        elif model_type == 'dual':
            print('Initialise Dual  model..')
            self.model = Dual(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity)
        elif model_type == 'dual_separate_input':
            print('Initialise Dual separate input model..')
            self.model = Dual_separate_input(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay, i_sim =self.i_sim,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity)
        elif model_type == 'dual_separate_input_inhibition':
            print('Initialise Dual separate input inhibition model..')
            self.model = Dual_separate_input_inhibition(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity, fb_connectivity=self.fb_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay, i_sim =self.i_sim,
                                     output_connectivity=self.output_connectivity)
        elif model_type == 'forward':
            print('Initialise Forward model..')
            self.model = Forward(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity)
        elif model_type == 'parallel':
            print('Initialise Parallel model..')
            self.model = Parallel(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                    eta=self.eta, beta=self.beta, decay=self.decay,
                                  output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity)

    def count_success(self, task, trial, model, choice, k):
        """
            Store the results of the final choice of the model: 1 if it made the right choice, 0 otherwise.
            parameters:
                task: class object
                model: class object
                trial: array of shape (n_cue, n_positions)
                       current trial of the task.
                    """
        self.legal_choices_array['session {}'.format(k)].append(int(task.is_legal_choice(trial, choice=choice)))
        self.all_trials['session {}'.format(k)]['all_trials'].append(trial)
        self.all_trials['session {}'.format(k)]['delay'].append(self.task.n_input_delay)
        self.all_trials['session {}'.format(k)]['input_time_1'].append(self.task.n_input_time_1)
        self.all_trials['session {}'.format(k)]['input_time_2'].append(self.task.n_input_time_2)
        if model.choice == task.get_best_choice(trial, self.deterministic, self.model.reward):
            self.success_array['session {}'.format(k)].append(1)
            if task.best_trial_first:
                self.success_array_best_first['session {}'.format(k)].append(1)
            else:
                self.success_array_best_last['session {}'.format(k)].append(1)

        else:
            self.success_array['session {}'.format(k)].append(0)
            if task.best_trial_first:
                self.success_array_best_first['session {}'.format(k)].append(0)
            else:
                self.success_array_best_last['session {}'.format(k)].append(0)

    def count_success_testing(self, task, trial, model, choice, k):
        """
            Store the results of the final choice of the model: 1 if it made the right choice, 0 otherwise.
            parameters:
                task: class object
                model: class object
                trial: array of shape (n_cue, n_positions)
                       current trial of the task.
                    """
        self.legal_choices_array_testing['session {}'.format(k)].append(int(task.is_legal_choice(trial, choice=choice)))
        self.all_trials_testing['session {}'.format(k)]['all_trials'].append(trial)
        self.all_trials_testing['session {}'.format(k)]['delay'].append(self.task.n_input_delay)
        self.all_trials_testing['session {}'.format(k)]['input_time_1'].append(self.task.n_input_time_1)
        self.all_trials_testing['session {}'.format(k)]['input_time_2'].append(self.task.n_input_time_2)
        if model.choice == task.get_best_choice(trial, self.deterministic, self.model.reward):
            self.success_array_testing['session {}'.format(k)].append(1)
            if task.best_trial_first:
                self.success_array_best_first_testing['session {}'.format(k)].append(1)
            else:
                self.success_array_best_last_testing['session {}'.format(k)].append(1)

        else:
            self.success_array_testing['session {}'.format(k)].append(0)
            if task.best_trial_first:
                self.success_array_best_first_testing['session {}'.format(k)].append(0)
            else:
                self.success_array_best_last_testing['session {}'.format(k)].append(0)

    def process_generalization_test(self, n_unseen=None):
        self.success_counter = 0
        n_test = 500
        trial_indexes = [i for i in range(self.task.unseen_trials.shape[0])]
        trials = self.task.unseen_trials
        if self.gen_test_indiv:
            print('Start generalization test on all 72 unknown pairs...')
            task_gen_test = Task(filename='json_files/task.json')
            for trial in task_gen_test.trials:
                self.process_one_trial(task_gen_test, trial, record_output=True)
                if self.model.choice == task_gen_test.get_best_choice(trial, self.deterministic, self.model.reward):
                    self.success_counter += 1
            self.success_indiv_gen_test['percent_success'].append(
                self.success_counter * 100 / len(task_gen_test.trials))
        else:
            for i in range(n_test):
                print(i)
                if int(i%len(trials)) == 0:
                    both = list(zip(trials, trial_indexes))
                    random.shuffle(both)
                    trials, trial_indexes = zip(*both)
                    trials, trial_indexes = np.array(trials), np.array(trial_indexes)
                trial = trials[int(i % len(trials))]
                self.process_one_trial(self.task, trial, record_output=True)
                if self.model.choice == self.task.get_best_choice(trial, self.deterministic, self.model.reward):
                    self.success_counter += 1
            self.success_gen_test['n_unseen {}'.format(n_unseen)]['percent_success'].append(
                self.success_counter*100/n_test)

    def record_results_gen_test(self):
        if self.gen_test_indiv:
            self.success_indiv_gen_test['mean'] = np.mean(self.success_indiv_gen_test['percent_success'])
            self.success_indiv_gen_test['std'] = np.std(self.success_indiv_gen_test['percent_success'])
            if self.save_gen_test:
                np.save(nextnonexistent(self.file_gen_test), self.success_indiv_gen_test)
        else:
            for n in self.n_unseen:
                self.success_gen_test['n_unseen {}'.format(n)]['mean'] = \
                    np.mean(self.success_gen_test['n_unseen {}'.format(n)]['percent_success'])
                self.success_gen_test['n_unseen {}'.format(n)]['std'] = \
                    np.std(self.success_gen_test['n_unseen {}'.format(n)]['percent_success'])
            if self.save_gen_test:
                np.save(nextnonexistent(self.file_gen_test), self.success_gen_test)


    def process_one_trial(self, task, trial, record_output=True, testing=False):
        self.count_record += 1
        self.model.record_output_activity[self.count_record] = {}
        self.model.record_output_activity[self.count_record]['output_activity'] = []
        self.model.record_output_activity[self.count_record]['trial_info'] = {}

        if self.model.separate_input:
            trial_with_chronogram_early, trial_with_chronogram_late = self.task.get_trial_with_chronogram_dual(trial)
            self.model.process(trial_with_chronogram_early, trial_with_chronogram_late, count_record=self.count_record,
                               record_output = record_output)
        else:
            trial_chronogram = task.get_trial_with_chronogram(trial)
            self.model.process(trial_chronogram, count_record=self.count_record, record_output=record_output)
        if task.best_trial_first:
            self.model.record_output_activity[self.count_record]['trial_info']['best_cue_first'] = True
        else:
            self.model.record_output_activity[self.count_record]['trial_info']['best_cue_first'] = False
        all_choices = [0, 1, 2, 3]
        if self.deterministic:
            for c in all_choices:
                self.model.record_output_activity[self.count_record]['trial_info'][c] = task.get_reward(trial, c,
                                                                                                        self.model.parameters['RL']['reward'],
                                                                                                        self.model.parameters['RL']['penalty'])
        else:
            legal_choices = task.get_legal_choices(trial)

            for c in legal_choices:
                self.model.record_output_activity[self.count_record]['trial_info'][c] = task.get_reward_probability(
                    trial, c)
                all_choices.remove(c)
            for c in all_choices:
                self.model.record_output_activity[self.count_record]['trial_info'][c] = 0


    def one_run(self, n_unseen=None):
        """
        Run the experiment n_sessions times. It is either a generalization test or a normal run.
        At each session, the trial list is first shuffled, before the model executes the task for each trial.
        parameters:
                - n_unseen: int
                  used when self.val_gen_test=True, corresponds to the number of unseen combinations that are not seen
                  by the model during training. Those unseen trials will be used only during the testing phase.
        """
        for k in range(self.n_sessions):
            self.init_model(seed=self.seeds[k], model_type=self.model_type)
            self.count_record = 0
            if self.val_gen_test and not self.gen_test_indiv:
                trials = self.task.get_random_trials(len(self.task.trials)-n_unseen)
            else:
                trials = self.task.trials
            trial_indexes = [i for i in range(trials.shape[0])]
            for i in range(self.nb_train):
                if self.temp_gen_test:
                    if self.overlap:
                        self.task.get_random_trial_length_overlap()
                    else:
                        self.task.get_random_trial_length_no_overlap()
                    #self.task.get_random_trial_length()
                    #self.task.get_random_trial_length_limited()
                    #self.task.get_random_trial_length_end_fixed()
                if int(i%len(trials)) == 0:
                    both = list(zip(trials, trial_indexes))
                    random.shuffle(both)
                    trials, trial_indexes = zip(*both)
                    trials, trial_indexes = np.array(trials), np.array(trial_indexes)
                trial = trials[int(i % len(trials))]
                self.process_one_trial(self.task, trial, record_output=True)
                reward = self.task.get_reward(trial, self.model.choice, self.model.reward, self.model.penalty)
                self.model.train(reward, self.model.choice)
                self.count_success(self.task, trial, self.model, self.model.choice, k)
            if self.val_gen_test:
                self.process_generalization_test(n_unseen)
            if self.testing:
                self.init_count_testing()
                print('Testing phase')
                if self.overlap:
                    self.task.get_random_trial_length_overlap()
                else:
                    self.task.get_random_trial_length_no_overlap()
                for i in range(1000):
                    if int(i % len(trials)) == 0:
                        both = list(zip(trials, trial_indexes))
                        random.shuffle(both)
                        trials, trial_indexes = zip(*both)
                        trials, trial_indexes = np.array(trials), np.array(trial_indexes)
                    trial = trials[int(i % len(trials))]
                    self.process_one_trial(self.task, trial, record_output=True, testing=True)
                    self.count_success_testing(self.task, trial, self.model, self.model.choice, k)
            self.result_sessions['session {}'.format(k)] = moving_average(self.success_array['session {}'.format(k)], 50)

    def run(self):
        if self.val_gen_test:
            if self.gen_test_indiv:
                print('Generalization test when model trained with individual cues')
                self.one_run(n_unseen=None)
            else:
                print('Generalization test with n_unseen= {}'.format(str(self.n_unseen)))
                for n in self.n_unseen:
                    self.one_run(n_unseen=n)
            self.record_results_gen_test()
        else:
            self.one_run()










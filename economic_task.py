import numpy as np
from math import comb, factorial
from itertools import permutations, combinations, product
import os
import json
import random
import math
import pylab as p
SEED = 10
random.seed(SEED)
np.random.seed(SEED)


class EconomicTask():
    def __init__(self, filename='json_files/evo_prospect/economic_task.json'):
        """Economic decision-making task. An agent chooses between two options.
         Each option corresponds to a pie chart composed of two slices.
         Each slice encodes one possible outcome of the lottery (x or 0). The arc length
         of each slice represents the probability of the corresponding outcome ( p or 1 − p).
         The relative positions on the pie chart of each slice are randomly determined at each trial.
        """
        self.filename = filename
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.setup()
        self.setup_control_trials()
        self.setup_tradeoff_trials()

    def setup(self):
        _ = self.parameters
        self.n_session = _["n_session"]
        self.nb_train = _["nb_train"]
        self.n_input_time_1 = _["n_input_time_1"]
        self.n_input_time_2 = _["n_input_time_2"]
        self.n_init_time = _["n_init_time"]
        self.n_end_time = _["n_end_time"]
        self.n_input_delay = _["n_input_delay"]
        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay
        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

    def setup_control_trials(self):
        """
        Group 1. There is a better response regardless of the risk attitude of the decision-maker:
         there is i, j ∈ {1, 2} s.t. pi ≥ pj and xi > xj, or pi > pj and xi ≥ xj (30 different pairs
         of lotteries for gains, 30 for losses ignoring the order/side of presentation—60 otherwise).
        """
        self.control_loss_lotteries = {}
        self.control_gain_lotteries = {}
        self.control_loss_lotteries['all_trials'] = []
        self.control_loss_lotteries['best_choices'] = []
        self.control_gain_lotteries['all_trials'] = []
        self.control_gain_lotteries['best_choices'] = []

        negative_x = list(combinations(range(-3, 0), 2))
        positive_x = list(combinations(range(1, 4), 2))
        all_p = list(combinations([0.25, 0.50, 0.75, 1], 2))
        for p in [0.25, 0.50, 0.75, 1]:
            all_p.append((p, p))
        for x in range(-3, 0):
            negative_x.append((x, x))
        for x in range(1, 4):
            positive_x.append((x, x))
        for x in negative_x:
            for p in all_p:
                if (x[0] <= x[1] and p[0] < p[1]) or (x[0] < x[1] and p[0] <= p[1]):
                    k = random.choice([0, 1])
                    if k == 0:
                        self.control_loss_lotteries['all_trials'].append([[x[1], p[1]], [x[0], p[0]]])
                        self.control_loss_lotteries['best_choices'].append(1)
                    else:
                        self.control_loss_lotteries['all_trials'].append([[x[0], p[0]], [x[1], p[1]]])
                        self.control_loss_lotteries['best_choices'].append(0)

        for x in positive_x:
            for p in all_p:
                if (x[0] <= x[1] and p[0] < p[1]) or (x[0] < x[1] and p[0] <= p[1]):
                    k = random.choice([0, 1])
                    if k == 0:
                        self.control_gain_lotteries['all_trials'].append([[x[1], p[1]], [x[0], p[0]]])
                        self.control_gain_lotteries['best_choices'].append(0)
                    else:
                        self.control_gain_lotteries['all_trials'].append([[x[0], p[0]], [x[1], p[1]]])
                        self.control_gain_lotteries['best_choices'].append(1)


        #### SHUFFLE inside the pairs

    def setup_tradeoff_trials(self):
        """
        Group 1. A trade-off between risk and potential gain/loss has to be made: there is i, j ∈ {1, 2}
         s.t. pi > pj and xi < xj (18 different pairs of lotteries for gains, 18 for losses).
        """
        self.tradeoff_gain_lotteries = {}
        self.tradeoff_gain_lotteries['all_trials'] = []
        self.tradeoff_gain_lotteries['riskiest_choices'] = []
        positive_x = list(permutations(range(1, 4), 2))
        all_p = list(combinations([0.25, 0.50, 0.75, 1.00], 2))
        for p in [0.25, 0.50, 0.75, 1.00]:
            all_p.append((p, p))
        for x in range(1, 4):
            positive_x.append((x, x))
        for x in positive_x:
            for p in all_p:
                if (0 < x[0] < x[1] and p[0] > p[1]) or (x[0] > x[1] > 0 and p[0] < p[1]):
                    k = random.choice([0, 1])
                    if k == 0:
                        self.tradeoff_gain_lotteries['all_trials'].append([[x[1], p[1]], [x[0], p[0]]])
                        self.tradeoff_gain_lotteries['riskiest_choices'].append(1)
                    else:
                        self.tradeoff_gain_lotteries['all_trials'].append([[x[0], p[0]], [x[1], p[1]]])
                        self.tradeoff_gain_lotteries['riskiest_choices'].append(0)

    def subjective_probability(self,p):
        """Subjective probability of X"""
        alpha = 0.0001
        return math.exp(-pow((-math.log(p)), alpha))


    def get_performance_assessment_trials(self):
        """
        Control 1: performance assessment. Lottery pairs with a better response (Group 1)
        are used to assess the monkeys’ performance. Here we consider specifically the cases
        where it exists i, j ∈ {1, 2}:
             — [Same p] pi = pj but xi > xj in order
             to assess the discrimination of the quantities (12 different pairs of lotteries
             for gains, 12 for losses ignoring the order/side of presentation—24 otherwise);
             — [Same x] xi = xj but pi > pj in order to assess the discrimination of the
             probabilities (18 different pairs of lotteries for gains, 18 for losses).
        """
        positive_lotteries = {}
        negative_lotteries = {}
        for group_lotteries in (positive_lotteries, negative_lotteries):
            for cond in ('same_p', 'same_x'):
                group_lotteries[cond] = {}
                for key in ('all_trials', 'best_choices'):
                    group_lotteries[cond][key] = []
        """positive_lotteries['same_p'] = {}
        positive_lotteries['same_p']['all_trials'] = []
        positive_lotteries['same_p']['best_choices'] = []
        positive_lotteries['same_x'] = {}
        positive_lotteries['same_x']['all_trials'] = []
        positive_lotteries['same_x']['best_choices'] = []
        
        negative_lotteries['same_p'] = {}
        negative_lotteries['same_p']['all_trials'] = []
        negative_lotteries['same_p']['best_choices'] = []
        negative_lotteries['same_x'] = {}
        negative_lotteries['same_x']['all_trials'] = []
        negative_lotteries['same_x']['best_choices'] = []"""
        for lottery in self.control_loss_lotteries['all_trials']:
            if lottery[0][0] == lottery[1][0]:
                negative_lotteries['same_x']['all_trials'].append(lottery)
                probs = [lottery[0][1], lottery[1][1]]
                negative_lotteries['same_x']['best_choices'].append(probs.index(max(probs)))
            if lottery[0][1] == lottery[1][1]:
                negative_lotteries['same_p']['all_trials'].append(lottery)
                x =[lottery[0][0], lottery[1][0]]
                negative_lotteries['same_p']['best_choices'].append(x.index(max(x)))
        for lottery in self.control_gain_lotteries['all_trials']:
            if lottery[0][0] == lottery[1][0]:
                positive_lotteries['same_x']['all_trials'].append(lottery)
                probs = [lottery[0][1], lottery[1][1]]
                positive_lotteries['same_x']['best_choices'].append(probs.index(max(probs)))
            if lottery[0][1] == lottery[1][1]:
                positive_lotteries['same_p']['all_trials'].append(lottery)
                x = [lottery[0][0], lottery[1][0]]
                positive_lotteries['same_p']['best_choices'].append(x.index(max(x)))
        return positive_lotteries, negative_lotteries

    def get_random_trials(self, gain, n=1):
        """ Get a random trial """
        if gain:
            trials = self.control_gain_lotteries['all_trials']
            best_choices = self.control_gain_lotteries['best_choices']
        else:
            trials = self.control_loss_lotteries['all_trials']
            best_choices = self.control_loss_lotteries['best_choices']

        if n == 1:
            index = np.random.randint(len(trials))
        else:
            index = random.sample(range(0, len(trials)), n)
            mask = np.ones(len(trials), dtype=bool)
            mask[index] = False
        return trials[index], best_choices[index]

    def chronogram(self, trial):
        schedule_1 = [(self.n_init_1, (0, 0)), (self.n_input_time_1,
                                                (trial[0][0]*trial[0][1], trial[0][0]*trial[0][1])), (self.n_end_1, (0, 0))]
        schedule_2 = [(self.n_init_2, (0, 0)),
                      (self.n_input_time_2, (trial[1][0]*trial[1][1], trial[1][0]*trial[1][1])), (self.n_end_2, (0, 0))]
        return np.concatenate([
            np.interp(np.arange(n), [0, n - 1], [beg, end])
            for (n, (beg, end)) in schedule_1]),  np.concatenate([
            np.interp(np.arange(n), [0, n - 1], [beg, end])
            for (n, (beg, end)) in schedule_2])

    def get_trial_with_chronogram(self, trial):
        trial_with_chronogram = self.chronogram(trial)
        return trial_with_chronogram

    def plot_chronogram(self, trial_with_chronogram):
        for i, v in enumerate(trial_with_chronogram):
            plt.subplot(2, 1, i + 1, frameon=False)
            plt.plot(v, lw=1)
            plt.yticks([0, max(v)])
            if i == (len(trial_with_chronogram)-1):
                plt.xticks([self.n_init_1, self.n_init_2,
                            self.n_init_2 + self.n_input_time_2 + self.n_end_2])
            else:
                plt.xticks([])
            plt.axvline(x=self.n_init_1, ymin = 0, ymax = 16, color='red', lw=1, alpha =0.5)
            plt.axvline(x=self.n_init_2, ymin=0, ymax=16, color='red', lw=1, alpha=0.5)
            plt.axvline(x=self.n_init_2 + self.n_input_time_2 + self.n_end_2,
                        ymin=0, ymax=16, color='red', lw=1, alpha=0.5)
        plt.ylim((0, 3))
        plt.tight_layout()
        plt.show()

    def get_reward(self, trial, choice):
        lottery_chosen = trial[choice]
        if np.random.random() <= lottery_chosen[1]:
            reward = lottery_chosen[0]
            proba = lottery_chosen[1]
        else:
            reward = 0
            proba = lottery_chosen[1]
        return reward, proba



    def get_best_choice(self, trial):
        return trial[1]




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    task = EconomicTask()
    positive_lotteries, negative_lotteries = task.get_performance_assessment_trials()

    task.setup_tradeoff_trials()
    #print(task.tradeoff_gain_lotteries)
    #print(len(task.tradeoff_gain_lotteries['all_trials']))


    for i, lottery in enumerate(task.control_gain_lotteries['all_trials']):
        best = task.control_gain_lotteries['best_choices'][i]
        if not lottery[best][1]*lottery[best][0] > lottery[1-best][1]*lottery[1-best][0]:
            print('error')

    """for elt in task.control_gain_lotteries['all_trials']:
        print(elt[0],elt[1])
        print('')"""
    #print(task.control_gain_lotteries)

    """for i in range(4):
        trial = task.get_random_trials(gain=True)
        print(trial)
        choice = np.random.choice([0,1])

        print('best choice', task.get_best_choice(trial))"""
        #print(task.get_reward(trial, choice=choice))













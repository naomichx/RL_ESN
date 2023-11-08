import numpy as np
from math import comb, factorial
from itertools import permutations, combinations, product
import os
import json
import random
import pylab as p
SEED = 10
random.seed(SEED)
np.random.seed(SEED)

def boards(p, q, n):
    comb = combinations(range(p), n)
    perm = permutations(range(q), n)
    coords = product(comb, perm)

    def make_board(c):
        arr = np.zeros((p, q), dtype=int)
        arr[c[0], c[1]] = 1
        return arr

    return map(make_board, coords)

def num_boards(p, q, n):
    return comb(p, n) * comb(q, n) * factorial(n)


class Task:
    def __init__(self, filename='task_test.json'):
        """
        This class implements a task where n stimuli (chosen among
        n_cue) are placed at n locations (chosen among n_pos). An
        agent
        has to pick a location and get the reward associated with
        the
        cue at that location. Each trial can hence be summarized
        by a
        (n_cue,n_pos) matrix with exactly two 1s inside (with no
        line nor column having more than one 1).

        Parameters:

          n_cue : int

            total number of cues

          n_pos : int

            total number of positions

          n_choice: int

            number of choices in a trial

          reward_probabilities : list

            Reward probability associated with each cue
        """
        self.filename = filename
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.setup()
        self.unseen_trials = None
        assert len(self.reward_probabilities) == self.n_cue

    def setup(self):
        _ = self.parameters
        self.n_session = _["n_session"]
        self.nb_train = _["nb_train"]
        self.n_cue = _["n_cue"]
        self.n_pos = _["n_position"]
        self.n_input_time_1 = _["n_input_time_1"]
        self.n_input_time_2 = _["n_input_time_2"]
        self.n_init_time = _["n_init_time"]
        self.n_end_time = _["n_end_time"]
        self.n_input_delay = _["n_input_delay"]
        self.n_choice = _["n_choice"]
        self.reward_probabilities = _["reward_probabilities"]
        self.best_trial_first = None
        self.trials = np.array([*boards(self.n_cue, self.n_pos, self.n_choice)])
        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay

        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

    def get_random_trial_length(self):
        #self.n_input_time_1 = np.random.choice([i for i in range(5, 10)])
        #self.n_input_time_2 = np.random.choice([i for i in range(5, 10)])

        self.n_input_time_1 = 5
        self.n_input_time_2 = 5
        self.n_input_delay = np.random.choice([i for i in range(0, 20)])
        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay
        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

    def get_random_trial_length_limited(self):
        self.n_input_time_1 = np.random.choice([i for i in range(5, 10)])
        self.n_input_delay = np.random.choice([i for i in range(0, 20)])
        self.n_input_time_2 = 30 - self.n_end_time - self.n_init_time - self.n_input_delay

        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay
        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

        #print(self.n_init_1 + self.n_input_time_1 + self.n_end_1)
        #print(self.n_init_2 + self.n_input_time_2 + self.n_input_delay + self.n_end_2)

    def get_random_trial_length_end_fixed(self):
        self.n_input_time_1 = np.random.choice([i for i in range(15, 25)])
        #self.n_input_delay = np.random.choice([i for i in range(8, self.n_input_time_1-3)])
        self.n_input_delay = np.random.choice([i for i in range(int(self.n_input_time_1/2),self.n_input_time_1-3 )])
        self.n_input_time_2 = self.n_input_time_1 - self.n_input_delay

        self.n_init_1 = 29 - self.n_input_time_1
        self.n_init_2 = 29 - self.n_input_time_2
        self.n_end_2 = self.n_end_time
        self.n_end_1 = self.n_end_time

        #print(self.n_init_1 + self.n_input_time_1 + self.n_end_1)
        #print(self.n_init_2 + self.n_input_time_2 + self.n_input_delay+ self.n_end_2)


        """if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)"""

    def get_random_trial_length_overlap(self):
        self.n_input_time_1 = np.random.choice([i for i in range(10, 18)])
        self.n_input_delay = np.random.choice([i for i in range(0, self.n_input_time_1-7)])
        #self.n_input_delay = np.random.choice([i for i in range(0, self.n_input_time_1 -4)])
        self.n_input_time_2 = 30 - self.n_end_time - self.n_init_time - self.n_input_delay

        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay
        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

    def get_random_trial_length_no_overlap(self):
        self.n_input_time_1 = np.random.choice([i for i in range(5, 10)])
        self.n_input_delay = np.random.choice([i for i in range(self.n_input_time_1, 20)])
        self.n_input_time_2 = 30 - self.n_end_time - self.n_init_time - self.n_input_delay

        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay
        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)



    def set_trial_delay(self, delay):
        self.n_input_delay = delay
        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay
        self.n_input_time_1 = 5
        self.n_input_time_2 = 5

        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

    def __getitem__(self, index):
        """ Get trial index """
        return self.trials[index]

    def __len__(self):
        """ Get number of trials """

        return len(self.trials)

    def get_random_trials(self, n=1):
        """ Get a random trial """
        if n == 1:
            index = np.random.randint(len(self))
        else:
            index = random.sample(range(0, len(self)), n)
            mask = np.ones(len(self), dtype=bool)
            mask[index] = False
            self.unseen_trials = self.trials[mask, ...]
        return self.trials[index]

    def separate_cues(self, trial):
        indexes = np.where(trial.sum(axis=1) == 1)[0]
        trial_with_best_cue = np.zeros((self.n_cue, self.n_pos))
        trial_with_worst_cue = np.zeros((self.n_cue, self.n_pos))
        trial_with_best_cue[indexes[0]] = trial[indexes[0]]
        trial_with_worst_cue[indexes[1]] = trial[indexes[1]]
        return trial_with_best_cue, trial_with_worst_cue

    def get_best_choice(self, trial, deterministic=False, reward = None):
        """Return the best choice for a given trial"""
        if deterministic:
            best_cue = np.argmax(trial.sum(axis=1) * reward)
            return int(np.where(trial[best_cue] == 1)[0])
        else:
            best_cue = np.argmax(trial.sum(axis=1) * self.reward_probabilities)
            return int(np.where(trial[best_cue] == 1)[0])

    def is_legal_choice(self, trial, choice):
        """Return whether choice is a legal choice for a given
        trial"""
        return trial.sum(axis=0)[choice] == 1

    def get_legal_choices(self, trial):
        """Return all legal choices for a given trial"""
        return np.nonzero(trial.sum(axis=0))[0]

    def get_reward_probability(self, trial, choice):
        """Return reward probability associated with a choice"""
        cue = np.argwhere(trial[:, choice] == 1)[0]
        return self.reward_probabilities[int(cue)]

    def get_reward(self, trial, choice, reward, penalty):
        """Return reward probability associated with a choice"""
        if self.is_legal_choice(trial, choice):
            p_reward = self.get_reward_probability(trial, choice)
            if random.random() < p_reward:
                return reward[np.argwhere(trial[:, choice] == 1)[0][0]]
            else:
                return 0
        else:
            return penalty   ## CORRECT

    def invert_probabilities(self):
        self.reward_probabilities = np.ones(self.n_cue) - self.reward_probabilities

    def chronogram(self):
        schedule_1 = [(self.n_init_1, (0, 0)), (self.n_input_time_1, (1, 1)), (self.n_end_1, (0, 0))]
        schedule_2 = [(self.n_init_2, (0, 0)), (self.n_input_time_2, (1, 1)), (self.n_end_2, (0, 0))]
        return np.concatenate([
            np.interp(np.arange(n), [0, n - 1], [beg, end])
            for (n, (beg, end)) in schedule_1]),  np.concatenate([
            np.interp(np.arange(n), [0, n - 1], [beg, end])
            for (n, (beg, end)) in schedule_2])

    def get_trial_with_chronogram_dual(self, trial):
        indexes = np.where(trial.sum(axis=1) == 1)[0]
        trial_with_best_cue, trial_with_worst_cue = self.separate_cues(trial)
        L1 = []
        L2 = []
        k = random.choice([0, 1])
        if k == 0:
            self.best_trial_first = True
            chrono_1 = self.chronogram()[0]
            chrono_2 = self.chronogram()[1]
        else:
            self.best_trial_first = False
            chrono_1 = self.chronogram()[1]
            chrono_2 = self.chronogram()[0]
        for i, v in enumerate(trial_with_best_cue.ravel()):
            L1.append(v * chrono_1)
        for j, v in enumerate(trial_with_worst_cue.ravel()):
            L2.append(v * chrono_2)
        trial_with_chronogram_1 = np.reshape(np.transpose(L1), (self.n_init_1 + self.n_input_time_1 +
                                                                       self.n_end_1, self.n_cue * self.n_pos))
        trial_with_chronogram_2 = np.reshape(np.transpose(L2), (self.n_init_1 + self.n_input_time_1 +
                                                                     self.n_end_1, self.n_cue * self.n_pos))
        if self.best_trial_first:
            return trial_with_chronogram_1, trial_with_chronogram_2
        else:
            return trial_with_chronogram_2, trial_with_chronogram_1

    def get_trial_with_chronogram(self, trial):
        indexes = np.where(trial.sum(axis=1) == 1)[0]
        L = []
        # plot_chronogram(trial, task)
        k = random.choice([0, 1])
        l = 1 - k
        if k == 0:
            self.best_trial_first = True
        else:
            self.best_trial_first = False
        for i, v in enumerate(trial[:indexes[0] + 1].ravel()):
            L.append(v * self.chronogram()[k])
        for j, v in enumerate(trial[indexes[0] + 1:].ravel()):
            L.append(v * self.chronogram()[l])
        trial_with_chronogram = np.transpose(L)
        trial_with_chronogram = np.reshape(trial_with_chronogram, (self.n_init_1 + self.n_input_time_1 +
                                            self.n_end_1, self.n_cue * self.n_pos))
        return trial_with_chronogram

    def plot_chronogram(self, trial_with_chronogram):

        to_plot = np.transpose(trial_with_chronogram)

        for i, v in enumerate(to_plot):
            plt.subplot(self.n_cue * self.n_pos, 1, i + 1, frameon=False)
            plt.plot(v, lw=1)
            plt.yticks([])

            if i == (len(to_plot)-1):
                plt.xticks([self.n_init_1, self.n_init_2,
                            self.n_init_2 + self.n_input_time_2 + self.n_end_2])
            else:
                plt.xticks([])

            plt.axvline(x=self.n_init_1, ymin = 0, ymax = 16, color='red', lw=1, alpha =0.5)
            plt.axvline(x=self.n_init_2, ymin=0, ymax=16, color='red', lw=1, alpha=0.5)
            plt.axvline(x=self.n_init_2 + self.n_input_time_2 + self.n_end_2,
                        ymin=0, ymax=16, color='red', lw=1, alpha=0.5)

        plt.tight_layout()
        plt.show()
#-----------------------------------------------------------------------------


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    task = Task(filename='json_files/stochastic/task_delay_11.json')


    """task.get_random_trial_length_limited()

    trial_with_chronogram_1, trial_with_chronogram_2 = task.get_trial_with_chronogram_dual(trial)

    print('shape', np.shape(trial_with_chronogram_1))
    print('shape 2', np.shape(trial_with_chronogram_2))

    print('trial', trial)
    print('best choice', task.get_best_choice(trial))
    print('best first', task.best_trial_first)"""

        #task.plot_chronogram(trial_with_chronogram_1)
        #task.plot_chronogram(trial_with_chronogram_2)

    #print('Find the new index of the selected trial:', np.where(trials==trial))
    #print('Find initial index:', trial_indexes[trials.index(trial)])

    #trial_with_chronogram = task.get_trial_with_chronogram(trial)

    #legal_choices = task.get_legal_choices(trial)
    #print(legal_choices)

    for i in range(10):
        task.get_random_trial_length_overlap()
        #print(task.n_input_time_1)
        #print(task.n_input_time_2)
        #print(task.n_input_delay)
        print('-------')


        trial = task.get_random_trials(n=1)
        choice = task.get_best_choice(trial, deterministic=False)
        print(task.get_reward_probability(trial, choice))
        #trial_with_chronogram = task.get_trial_with_chronogram(trial)
        #task.plot_chronogram(trial_with_chronogram)

    #print('best:', best_c)
    #print('worst:', worst_c)

    #for i, v in enumerate(trial.ravel()):
     #   print(v)
      #  print('chrono', v*task.chronogram())




    #print('Number of trials:', len(task.trials))

    # Legal choices for this trial
    #choices = task.get_legal_choices(trial)

    # Best choice for this trial
    #best_choice = task.get_best_choice(trial)
    #print('Best choice', best_choice)


import scipy.sparse as sp
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, Input, RLS
from reservoirpy.mat_gen import Initializer, uniform, random_sparse, normal, bernoulli
from reservoirpy.activationsfunc import sigmoid
rpy.verbosity(0)
import numpy as np
import json
import os
from utils import W_initializer_SW_1000, W_initializer_1000, W_initializer_cluster_scale_free_1000
import random
import math
random.seed(1)
np.random.seed(1)


def subjective_utility(X):
    """ Subjective utility of X """
    lambd = 1.
    rho = 0.1
    return np.where(X > 0, pow(abs(X), rho), -lambd * pow(abs(X), rho))

def subjective_probability(p):
    """Subjective probability of X"""
    alpha = 0.01
    return math.exp(-pow((-math.log(p)), alpha))


class Regular:
    def __init__(self, seed, filename='model.json', hyperparam_optim=False, units=None, lr=None, sr=None,
                 input_scaling=None, rc_connectivity=None, noise_rc=None, fb_scaling=None, input_connectivity=None,
                 output_connectivity=None, eta=None, beta=None, r_th=None, reward=None, fb_connectivity=None,
                 penalty=None, decay=None):
        """
        This class implements the Echo State Network Model trained
        with online RL.

        parameters:

                units: int
                        number of reservoir neurons
                sr: float
                        spectral radius
                lr: float
                        leak rate
                fb_scaling: float
                        feedback scaling
                input_scaling: float
                        input scaling
                noise_rc: float
                        reservoir noise
                rc_connectivity: float
                        reservoir connectivity
                input_connectivity: float
                        input connectivity
                fb_connectivity: float
                        feedback connectivity

                beta: int
                      inverse temperature
                eta: float
                    learning rate of the RL model
                r_th: float

        """

        self.filename = filename
        self.seed = seed
        self.n_lotteries = 2

        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)

        self.setup(hyperparam_optim=hyperparam_optim, units=units, lr=lr, sr=sr, input_scaling=input_scaling,
                   rc_connectivity=rc_connectivity, noise_rc=noise_rc, fb_scaling=fb_scaling,
                   input_connectivity=input_connectivity, output_connectivity=output_connectivity, eta=eta, beta=beta,
                   r_th=r_th, reward=reward, fb_connectivity=fb_connectivity, penalty=penalty,
                   decay=decay)
        self.all_p = None
        self.esn_output = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.V = [0.5 for i in range(72)]


    def setup(self, hyperparam_optim=False, units=None, lr=None, sr=None, input_scaling=None, rc_connectivity=None,
              noise_rc=None, fb_scaling=None, input_connectivity=None, output_connectivity=None, eta=None, beta=None,
              r_th=None, reward=None, fb_connectivity=None, penalty=None, LTP=None, LTD=None,decay=None):

        _ = self.parameters
        self.topology = _['ESN']['topology']
        if hyperparam_optim:
            self.units = _['ESN']['n_units']
            self.r_th = _['RL']['r_th']
            self.reward = _['RL']['reward']
            self.penalty = _['RL']['penalty']
            self.noise_rc = _['ESN']['noise_rc']
            self.fb_scaling = _['ESN']['fb_scaling']
            self.fb_connectivity = _['ESN']['fb_connectivity']
            self.input_scaling = _['ESN']['input_scaling']
            self.output_connectivity = _['ESN']['output_connectivity']
            self.lr = lr[0]
            self.sr = sr[0]
            self.rc_connectivity = rc_connectivity[0]
            self.input_connectivity = input_connectivity[0]
            self.eta = eta
            self.beta = beta
            self.decay = decay
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.r_th = _['RL']['r_th']
            self.reward = _['RL']['reward']
            self.penalty = _['RL']['penalty']
            self.decay = _['RL']['decay']
            self.units = _['ESN']['n_units']
            self.lr =_['ESN']['lr']
            self.sr = _['ESN']['sr']
            self.rc_connectivity = _['ESN']['rc_connectivity']
            self.noise_rc = _['ESN']['noise_rc']
            self.fb_scaling = _['ESN']['fb_scaling']
            self.input_connectivity = _['ESN']['input_connectivity']
            self.fb_connectivity = _['ESN']['fb_connectivity']
            self.input_scaling = _['ESN']['input_scaling']
            self.output_connectivity = _['ESN']['output_connectivity']

        #self.W = normal(loc=0,
         #                    scale=self.sr ** 2 / (self.rc_connectivity * self.units),
          #                   seed=self.seed)


        self.reservoir = Reservoir(units=self.units, lr=self.lr, sr=self.sr,
                                   input_scaling=self.input_scaling, W=uniform(low=-1, high=1),
                                   rc_connectivity=self.rc_connectivity, noise_rc=self.noise_rc,
                                   fb_scaling=self.fb_scaling,
                                   input_connectivity=self.input_connectivity,
                                   fb_connectivity=self.fb_connectivity, seed=self.seed, activation='tanh')
        self.readout = Ridge(self.n_lotteries,
                             Wout=uniform(low=-1, high=1, connectivity=self.output_connectivity, seed=self.seed))
        self.reservoir <<= self.readout
        self.esn = self.reservoir >> self.readout
        np.random.seed(self.seed)
        random.seed(self.seed)


    def softmax(self, x):
        """
            Return the softmax of x corresponding to probabilities, the sum of all
            probabilities is equal to 1.

            parameters:
                x: array of shape (n_output,)

                    """
        all_p = np.exp(self.beta * x)
        index_inf = None
        for i in range(len(all_p)):
            if np.isinf(all_p[i]):
                index_inf = i
        if index_inf is not None:
            all_p = [0 for i in range(self.n_lotteries)]
            all_p[index_inf] = 1
        elif all(k == 0 for k in list(all_p)):
            index_max = np.argmax(x)
            all_p = [0 for i in range(self.n_lotteries)]
            all_p[index_max] = 1
        else:
            all_p = [all_p[i] / np.sum(np.exp(self.beta * x), axis=0) for i in range(self.n_lotteries)]
        return all_p

    def softmax_neg(self, x):
        """
            Return the softmax of x corresponding to probabilities, the sum of all
            probabilities is equal to 1.

            parameters:
                x: array of shape (n_output,)

                    """
        all_p = np.exp(self.beta * x)
        index_inf = None
        for i in range(len(all_p)):
            if np.isinf(all_p[i]):
                index_inf = i
        if index_inf is not None:
            all_p = [0 for i in range(self.n_lotteries)]
            all_p[index_inf] = 1
        elif all(k == 0 for k in list(all_p)):
            index_max = np.argmax(x)
            all_p = [0 for i in range(self.n_lotteries)]
            all_p[index_max] = 1
        else:
            all_p = [-all_p[i] / np.sum(np.exp(self.beta * x), axis=0) for i in range(self.n_lotteries)]
        return all_p

    def select_choice(self):
        """
            Compute the choice of the ESN model
            """
        p = np.random.random()
        if p < self.epsilon:
            self.choice = np.random.choice(2)
        else:
            self.choice = np.argmax(self.esn_output)
        self.epsilon *= self.decay

    def process(self, trial_chronogram, count_record=None, record_output= False):
        for i in range(len(trial_chronogram)):
            self.esn_output = self.esn.run(trial_chronogram[i].ravel())[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout.state()[0])
        self.select_choice()

    def train(self, reward, proba, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = np.array(self.readout.params['Wout'].todense())
        else:
            W_out = np.array(self.readout.params['Wout'])
        if self.flag:
            self.mask = W_out != 0
            self.flag = False
        r = self.reservoir.state()
        #print('w*u', subjective_utility(reward) * subjective_probability(proba))
        #print('output',self.esn_output)
        #print('max',max(self.esn_output))
        #print('mean', np.mean(self.esn_output))
        #print('output/max', self.esn_output[choice]/max(self.esn_output))
        #print('rpe', subjective_utility(reward) * subjective_probability(proba) -
        #                                         self.esn_output[choice]/max(self.esn_output))
        EV_esn = self.esn_output[choice]#-min(self.esn_output)) #/ (max(self.esn_output)-min(self.esn_output))
        #print(self.esn_output)
        #print('reward', reward)
        #print('proba', proba)

        EV = subjective_utility(reward) * subjective_probability(proba)
        #print('EV', EV)
        EV_min = -3 * subjective_probability(1)
        EV_max = 0
        EV_norm = (EV_max - EV_min)/(EV - EV_min)
        #print('ev esn', EV_esn)
        #print('ev', EV)
        #print('ev diff', EV-EV_esn)

        # GAIN CASE
        EV = subjective_utility(reward)# * subjective_probability(proba)
        EV=reward
        EV_min = 0
        EV_max = subjective_utility(3) * subjective_probability(1)
        EV_norm = (EV - EV_min) / (EV_max - EV_min)
        W_out[:, choice] += np.array(self.eta * (EV - EV_esn) * (r[0][:] - self.r_th))
        W_out = W_out*self.mask
        self.all_W_out.append(W_out)
        for i in range(self.n_lotteries):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)

    def train_meropi(self, reward, proba, choice, trial_index):
        if sp.issparse(self.readout.params['Wout']):
            W_out = np.array(self.readout.params['Wout'].todense())
        else:
            W_out = np.array(self.readout.params['Wout'])
        if self.flag:
            self.mask = W_out != 0
            self.flag = False
        r = self.reservoir.state()
        W_max = 1
        W_min = 0
        EV_esn = (self.esn_output[choice] -min(self.esn_output))/ (max(self.esn_output)-min(self.esn_output))
        EV = subjective_utility(reward) * subjective_probability(proba)
        #EV_esn = self.esn_output[choice]
        EV_min = 0
        EV_max = subjective_utility(3) * subjective_probability(1)
        EV_norm = (EV - EV_min) / (EV_max - EV_min)

        #RPE = reward - self.V[trial_index]
        RPE = EV_norm-EV_esn
        #self.V[trial_index] += self.eta * RPE

        delta_W_out = self.eta * RPE * r[0][:]
        W_out[:, choice] += np.array(delta_W_out * W_out[:, choice])
        W_out = W_out * self.mask
        self.all_W_out.append(W_out)
        self.readout.params['Wout'] = sp.csr_matrix(W_out)
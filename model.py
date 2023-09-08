import scipy.sparse as sp
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, Input, RLS
from reservoirpy.mat_gen import Initializer, uniform, random_sparse, normal
rpy.verbosity(0)
import numpy as np
import json
import os
from utils import W_initializer_SW_1000, W_initializer_1000, W_initializer_cluster_scale_free_1000
import random
random.seed(1)
np.random.seed(1)


def threshold_like(x: np.ndarray) -> np.ndarray:
    """Threshold-like activation function.

    .. math::

        y_k =[a / (b+ exp(-k(z-c))] - d

        a = 1
        b = 1
        c = 1
        k = 10
        d = 0

    Parameters
    ----------
    x : array
        Input array.
    beta: float, default to 1.0
        Beta parameter of softmax.
    Returns
    -------
    array
        Activated vector.
    """
    a, b, c, k, d = 1, 1, 1, 10, 0
    _x = np.asarray(x)
    return (a/(b+np.exp(-k*(_x-c))))-d


class Regular:
    def __init__(self, seed, filename='model.json', n_position=4, hyperparam_optim=False, units=None, lr=None, sr=None,
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

        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
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



    def setup(self, hyperparam_optim=False, units=None, lr=None, sr=None, input_scaling=None, rc_connectivity=None,
              noise_rc=None, fb_scaling=None, input_connectivity=None, output_connectivity=None, eta=None, beta=None,
              r_th=None, reward=None, fb_connectivity=None, penalty=None, LTP=None, LTD=None,decay=None):

        _ = self.parameters
        self.topology = _['ESN']['topology']
        self.activation_func = _['ESN']['activation']


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

        self.W = normal(loc=0,
                             scale=self.sr ** 2 / (self.rc_connectivity * self.units),
                             seed=self.seed)


        print('Activation function direct pathway:', self.activation_func)
        self.reservoir = Reservoir(units=self.units, lr=self.lr, sr=self.sr,
                                   input_scaling=self.input_scaling,
                                   W=self.W,
                                   rc_connectivity=self.rc_connectivity, noise_rc=self.noise_rc,
                                   fb_scaling=self.fb_scaling,
                                   input_connectivity=self.input_connectivity,
                                   fb_connectivity=self.fb_connectivity, seed=self.seed)
        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))
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
            all_p = [0 for i in range(self.n_position)]
            all_p[index_inf] = 1
        elif all(k == 0 for k in list(all_p)):
            index_max = np.argmax(x)
            all_p = [0 for i in range(self.n_position)]
            all_p[index_max] = 1
        else:
            all_p = [all_p[i] / np.sum(np.exp(self.beta * x), axis=0) for i in range(self.n_position)]
        return all_p

    def select_choice(self):
        """
            Compute the choice of the ESN model
            """
        p = np.random.random()
        if p < self.epsilon:
            self.choice = np.random.choice(4)
        else:
            self.choice = np.argmax(self.esn_output)
        self.epsilon *= self.decay


    def process(self, trial_chronogram, count_record=None, record_output= False):
        for i in range(len(trial_chronogram)):
            self.esn_output = self.esn.run(trial_chronogram[i].ravel())[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout.state()[0])
        self.select_choice()

    def train(self, reward, choice):
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
        W_out[:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r[0][:] - self.r_th))
        W_out = W_out*self.mask
        self.all_W_out.append(W_out)

        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class Dual(Regular):
    def __init__(self, seed, filename='wide_model.json', n_position=4, hyperparam_optim=False, lr=None, sr=None,
                  rc_connectivity=None, input_connectivity=None, eta=None, beta=None, decay=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim,  lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,
                   eta= eta, beta=beta, decay=decay)

        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1

    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None):

        _ = self.parameters

        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.activation_func = _['ESN_1']['activation']

        if self.activation_func == 'tanh':
            activation_func = 'tanh'
        elif self.activation_func == 'threshold':
            activation_func = threshold_like

        self.output_connectivity = _['Readout']['output_connectivity']
        self.units = _['ESN_1']['n_units']
        self.noise_rc = _['ESN_1']['noise_rc']
        self.fb_scaling = _['ESN_1']['fb_scaling']
        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            for i in range(2):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            for i in range(2):
                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i + 1))]['rc_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i + 1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))

        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), 2)

        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))

        self.reservoir = {}
        for i in range(2):
            self.reservoir[i] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=seeds[i],
                                          activation=activation_func)
            self.reservoir[i] <<= self.readout
        self.esn = [self.reservoir[0], self.reservoir[1]] >> self.readout
        np.random.seed(self.seed)
        random.seed(self.seed)

    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']

        if self.flag:
            self.mask = np.array(W_out != 0)
            self.flag = False

        r_states = {}
        for i in range(2):
            r_states[i] = self.reservoir[i].state()

        W_out_dict = {}
        for i in range(2):
            W_out_dict[i] = np.array(W_out[i * self.units:(i + 1) * self.units])
            W_out_dict[i][:, choice] += np.array(
                self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r_states[i][0][:] - self.r_th))
        W_out = np.concatenate((W_out_dict[0], W_out_dict[1]))
        W_out = W_out * self.mask
        self.all_W_out.append(W_out)

        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class Converging(Regular):
    def __init__(self, seed, filename, n_position=4,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay= None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,eta=eta, beta=beta, decay=decay)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1


    def setup(self, hyperparam_optim=False,  lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None):

        _ = self.parameters
        self.n_reservoir = 3

        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.activation_func = _['ESN_1']['activation']


        if self.activation_func == 'tanh':
            activation_func = 'tanh'
        elif self.activation_func == 'threshold':
            activation_func = threshold_like

        self.output_connectivity = _['Readout']['output_connectivity']

        self.units = _['ESN_1']['n_units']
        self.noise_rc = _['ESN_1']['noise_rc']
        self.fb_scaling = _['ESN_1']['fb_scaling']
        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            for i in range(self.n_reservoir):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                    scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                    seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            for i in range(self.n_reservoir):
                self.lr.append(_['ESN_{}'.format(str(i+1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i+1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i+1))]['rc_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i+1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1]** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))

        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), self.n_reservoir)

        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))

        self.reservoir  = {}
        for i in range(self.n_reservoir):
            self.reservoir[i] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=self.seed,
                                          activation=activation_func)

        self.reservoir[0] <<= self.readout
        self.reservoir[1] <<= self.readout

        self.esn = [self.reservoir[0], self.reservoir[1],
                    [self.reservoir[0], self.reservoir[1]] >> self.reservoir[2]] >> self.readout




    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']

        if self.flag:
            self.mask = np.array(W_out != 0)
            self.flag = False

        r_states = {}
        for i in range(self.n_reservoir):
            r_states[i] = self.reservoir[i].state()

        W_out_dict = {}
        for i in range(self.n_reservoir):
            W_out_dict[i] = np.array(W_out[i*self.units:(i+1)*self.units])
            W_out_dict[i][:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r_states[i][0][:] - self.r_th))



        W_out = np.concatenate(tuple([W_out_dict[i] for i in range(self.n_reservoir)]))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class Differential(Regular):
    def __init__(self, seed, filename, n_position=4,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity, input_connectivity=input_connectivity,
                   eta=eta, beta=beta, decay=decay)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1

    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None):

        _ = self.parameters

        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.activation_func = _['ESN_1']['activation']

        if self.activation_func == 'tanh':
            activation_func = 'tanh'
        elif self.activation_func == 'threshold':
            activation_func = threshold_like

        self.output_connectivity = _['Readout']['output_connectivity']
        self.units = _['ESN_1']['n_units']
        self.noise_rc = _['ESN_1']['noise_rc']
        self.fb_scaling = _['ESN_1']['fb_scaling']
        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            for i in range(3):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            for i in range(3):
                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i + 1))]['rc_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i + 1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))

        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), 3)

        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))

        self.reservoir = {}
        for i in range(3):
            self.reservoir[i] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=seeds[i],
                                          activation=activation_func)
        self.reservoir[0] <<= self.readout
        self.reservoir[2] <<= self.readout

        self.esn = [self.reservoir[0], self.reservoir[0]>>self.reservoir[1], self.reservoir[2]] >> self.readout
        np.random.seed(self.seed)
        random.seed(self.seed)

    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']

        if self.flag:
            self.mask = np.array(W_out != 0)
            self.flag = False

        r_states = {}
        for i in range(3):
            r_states[i] = self.reservoir[i].state()

        W_out_dict = {}
        for i in range(3):
            W_out_dict[i] = np.array(W_out[i*self.units:(i+1)*self.units])
            W_out_dict[i][:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r_states[i][0][:] - self.r_th))


        W_out = np.concatenate((W_out_dict[0], W_out_dict[1], W_out_dict[2]))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)



class Forward(Regular):
    def __init__(self, seed, filename='deep_deep_model.json', n_position=4,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity, eta=eta, beta=beta, decay=decay)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1

    def setup(self, hyperparam_optim=False,  lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None):

        _ = self.parameters

        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.activation_func = _['ESN_1']['activation']

        if self.activation_func == 'tanh':
            activation_func = 'tanh'
        elif self.activation_func == 'threshold':
            activation_func = threshold_like
        self.output_connectivity = _['Readout']['output_connectivity']
        self.units = _['ESN_1']['n_units']
        self.noise_rc = _['ESN_1']['noise_rc']
        self.fb_scaling = _['ESN_1']['fb_scaling']
        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            for i in range(3):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                    scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                    seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            for i in range(3):
                self.lr.append(_['ESN_{}'.format(str(i+1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i+1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i+1))]['rc_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i+1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1]** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))

        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), 3)

        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))

        self.reservoir  = {}
        for i in range(3):
            self.reservoir[i] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=seeds[i],
                                          activation=activation_func)
            self.reservoir[i] <<= self.readout

        self.esn = [self.reservoir[0], self.reservoir[1],
                    self.reservoir[0] >> self.reservoir[1] >> self.reservoir[2]] >> self.readout


    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']

        if self.flag:
            self.mask = np.array(W_out != 0)
            self.flag = False

        r_states = {}
        for i in range(3):
            r_states[i] = self.reservoir[i].state()

        W_out_dict = {}
        for i in range(3):
            W_out_dict[i] = np.array(W_out[i*self.units:(i+1)*self.units])
            W_out_dict[i][:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r_states[i][0][:] - self.r_th))


        W_out = np.concatenate((W_out_dict[0], W_out_dict[1], W_out_dict[2]))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)



class Parallel(Regular):
    def __init__(self, seed, filename='deep_deep_model.json', n_position=4,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None,decay=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,eta=eta, beta=beta, decay=decay)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1

    def setup(self, hyperparam_optim=False,  lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay= None):

        _ = self.parameters
        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.activation_func = _['ESN_1']['activation']
        if self.activation_func == 'tanh':
            activation_func = 'tanh'
        elif self.activation_func == 'threshold':
            activation_func = threshold_like

        self.output_connectivity = _['Readout']['output_connectivity']

        self.units = _['ESN_1']['n_units']
        self.noise_rc = _['ESN_1']['noise_rc']
        self.fb_scaling = _['ESN_1']['fb_scaling']
        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            for i in range(3):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                    scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                    seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            for i in range(3):
                self.lr.append(_['ESN_{}'.format(str(i+1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i+1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i+1))]['rc_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i+1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1]** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))

        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), 3)

        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))

        self.reservoir  = {}
        for i in range(3):
            self.reservoir[i] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=seeds[i],
                                          activation=activation_func)
            self.reservoir[i] <<= self.readout

        self.esn = [self.reservoir[0], self.reservoir[1], self.reservoir[2]] >> self.readout


    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']

        if self.flag:
            self.mask = np.array(W_out != 0)
            self.flag = False

        r_states = {}
        for i in range(3):
            r_states[i] = self.reservoir[i].state()

        W_out_dict = {}
        for i in range(3):
            W_out_dict[i] = np.array(W_out[i*self.units:(i+1)*self.units])
            W_out_dict[i][:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r_states[i][0][:] - self.r_th))


        W_out = np.concatenate((W_out_dict[0], W_out_dict[1], W_out_dict[2]))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)

























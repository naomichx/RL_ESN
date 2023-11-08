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

def negative_sigmoid(x: np.ndarray) -> np.ndarray:
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

    return -sigmoid(x)


class Regular:
    def __init__(self, seed, filename='model.json', n_position=4, hyperparam_optim=False, units=None, lr=None, sr=None,
                 input_scaling=None, rc_connectivity=None, noise_rc=None, fb_scaling=None, input_connectivity=None,
                  eta=None, beta=None, r_th=None, reward=None, fb_connectivity=None,output_connectivity=None,
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
                   input_connectivity=input_connectivity, eta=eta, beta=beta,
                   r_th=r_th, reward=reward, fb_connectivity=fb_connectivity, output_connectivity=output_connectivity,
                   penalty=penalty, decay=decay)
        self.all_p = None
        self.esn_output = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = False




    def setup(self, hyperparam_optim=False, units=None, lr=None, sr=None, input_scaling=None, rc_connectivity=None,
              noise_rc=None, fb_scaling=None, input_connectivity=None, output_connectivity=None, eta=None, beta=None,
              r_th=None, reward=None, fb_connectivity=None, penalty=None,decay=None):

        _ = self.parameters
        self.topology = _['ESN']['topology']
        #self.activation_func = _['ESN']['activation']



        self.activation_func = 'tanh'
        if self.activation_func == 'tanh':
            activation_func = 'tanh'
        elif self.activation_func == 'threshold':
            activation_func = threshold_like
        print('Activation function:', self.activation_func)


        if hyperparam_optim:
            self.units = _['ESN']['n_units']
            self.r_th = _['RL']['r_th']
            self.reward = _['RL']['reward']
            self.penalty = _['RL']['penalty']
            self.noise_rc = _['ESN']['noise_rc']
            self.fb_scaling = _['ESN']['fb_scaling']
            self.fb_connectivity = fb_connectivity[0]
            self.input_scaling = _['ESN']['input_scaling']
            self.output_connectivity = output_connectivity
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
                                   fb_connectivity=self.fb_connectivity, seed=self.seed,
                                   activation= activation_func)
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
            all_p = [0 for i in range(len(all_p))]
            all_p[index_inf] = 1
        elif all(k == 0 for k in list(all_p)):
            index_max = np.argmax(x)
            all_p = [0 for i in range(len(all_p))]
            all_p[index_max] = 1
        else:
            all_p = [all_p[i] / np.sum(np.exp(self.beta * x), axis=0) for i in range(len(all_p))]
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
            self.esn_output = self.esn.call(trial_chronogram[i].ravel())[0]
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
                  rc_connectivity=None, input_connectivity=None, eta=None, beta=None, decay=None,
                 fb_connectivity=None,output_connectivity=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim,  lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,
                   eta=eta, beta=beta, decay=decay, fb_connectivity=fb_connectivity,
                   output_connectivity=output_connectivity)

        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = False
        self.activation_func == 'tanh'
        print('Activation:', self.activation_func)

    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None,fb_connectivity=None,output_connectivity=None,):

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

        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.fb_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(2):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.fb_connectivity.append(fb_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            self.output_connectivity = _['Readout']['output_connectivity']
            for i in range(2):
                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
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
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
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


class Dual_separate_input(Regular):
    def __init__(self, seed, filename='wide_model.json', n_position=4, hyperparam_optim=False, lr=None, sr=None,
                  rc_connectivity=None, input_connectivity=None, eta=None, beta=None, decay=None, i_sim=None,
                 fb_connectivity=None, output_connectivity=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim,  lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,
                   eta= eta, beta=beta, decay=decay, i_sim=i_sim, fb_connectivity=fb_connectivity, output_connectivity=output_connectivity)

        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = True
        print('Separate input:', self.separate_input)

    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None, i_sim=None,fb_connectivity=None,output_connectivity=None,):

        _ = self.parameters

        self.i_sim =i_sim
        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']

        self.activation_func = 'tanh'
        print('Activation:', self.activation_func)

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
        self.fb_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(2):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])

                self.fb_connectivity.append(fb_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            self.output_connectivity = _['Readout']['output_connectivity']
            for i in range(2):

                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i + 1))]['rc_connectivity'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
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
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
                                          activation=activation_func,
                                          name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(i)))
            self.reservoir[i] <<= self.readout
        self.esn = [self.reservoir[0], self.reservoir[1]] >> self.readout
        np.random.seed(self.seed)
        random.seed(self.seed)

    def process(self, trial_chronogram_early, trial_chronogram_late, count_record=None, record_output=False):
        for i in range(len(trial_chronogram_early)):
            self.esn_output = self.esn.call({'reservoir_{}'.format(self.i_sim) + '_0': trial_chronogram_early[i].ravel(),
                                            'reservoir_{}'.format(self.i_sim) + '_1': trial_chronogram_late[i].ravel()})[0]
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
                 eta=None, beta=None, decay= None,fb_connectivity=None,output_connectivity=None,):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,eta=eta, beta=beta, decay=decay, fb_connectivity=fb_connectivity,
                   output_connectivity=output_connectivity)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = False


    def setup(self, hyperparam_optim=False,  lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None,fb_connectivity=None,output_connectivity=None):

        _ = self.parameters
        self.n_reservoir = 3

        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        #self.activation_func = _['ESN_1']['activation']

        self.activation_func = 'tanh'
        print('Activation:', self.activation_func)


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
        self.fb_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity=output_connectivity
            for i in range(self.n_reservoir):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.fb_connectivity.append(fb_connectivity[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                    scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                    seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            self.output_connectivity = _['Readout']['output_connectivity']
            for i in range(self.n_reservoir):
                self.lr.append(_['ESN_{}'.format(str(i+1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i+1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i+1))]['rc_connectivity'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
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
                                          fb_connectivity=self.fb_connectivity[i], seed=self.seed,
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
                 eta=None, beta=None, decay=None,fb_connectivity=None,output_connectivity=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity, input_connectivity=input_connectivity,
                   eta=eta, beta=beta, decay=decay, fb_connectivity=fb_connectivity, output_connectivity=output_connectivity)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = False

    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None,fb_connectivity=None,output_connectivity=None,):

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
        self.fb_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(3):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.fb_connectivity.append(fb_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            self.output_connectivity = _['Readout']['output_connectivity']
            for i in range(3):
                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i + 1))]['rc_connectivity'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
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
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
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
    def __init__(self, seed, filename, n_position=4,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay=None, fb_connectivity=None, output_connectivity=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity, eta=eta, beta=beta, decay=decay,
                   fb_connectivity=fb_connectivity, output_connectivity=output_connectivity)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = False

    def setup(self, hyperparam_optim=False,  lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None, fb_connectivity=None,output_connectivity=None):

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
        self.fb_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(3):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.fb_connectivity.append(fb_connectivity[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                    scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                    seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            self.output_connectivity =  _['Readout']['output_connectivity']
            for i in range(3):
                self.lr.append(_['ESN_{}'.format(str(i+1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i+1))]['sr'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
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
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
                                          activation=activation_func)
            self.reservoir[i] <<= self.readout

        #self.reservoir[0] <<= self.reservoir[1]
        #self.reservoir[1] <<= self.reservoir[2]

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
                 eta=None, beta=None,decay=None,fb_connectivity=None,output_connectivity=None, separate_input=False):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,eta=eta, beta=beta, decay=decay,
                   fb_connectivity=fb_connectivity, output_connectivity=output_connectivity)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = separate_input

    def setup(self, hyperparam_optim=False,  lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay= None,fb_connectivity=None,output_connectivity=None):

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
        self.fb_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(3):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.fb_connectivity.append(fb_connectivity[i])
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
                self.fb_connectivity.append(_['ESN_{}'.format(str(i+1))]['fb_connectivity'])
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
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
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





class Dual_separate_input_(Regular):
    def __init__(self, seed, filename='wide_model.json', n_position=4, hyperparam_optim=False, lr=None, sr=None,
                  rc_connectivity=None, input_connectivity=None, eta=None, beta=None, decay=None, i_sim=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim,  lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,
                   eta= eta, beta=beta, decay=decay, i_sim=i_sim)

        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = [True,True,True]
        self.mask = {}
        self.epsilon = 1
        self.separate_input = True
        print('Separate input:', self.separate_input)

    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None, i_sim=None):

        _ = self.parameters

        self.i_sim =i_sim

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
        seeds = random.sample(range(1, 100), 3)
        self.readout = {}
        self.readout[0] = Ridge(16,
                                Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=seeds[0]))
        self.readout[1] = Ridge(16,
                                Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=seeds[1]))
        self.readout[2] = Ridge(4,
                                Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=seeds[1]))
        self.reservoir = {}
        for i in range(2):
            self.reservoir[i] = Reservoir(units=250, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=seeds[i],
                                          activation='tanh',
                                          name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(i)))

        part_0 = self.reservoir[0] >> self.readout[0]
        part_1 = self.reservoir[1] >> self.readout[1]
        #self.esn = [part_0, part_1] >> self.reservoir[2] >> self.readout[2]
        self.esn = [part_0, part_1] >> self.readout[2]
        np.random.seed(self.seed)
        random.seed(self.seed)

    def process(self, trial_chronogram_early, trial_chronogram_late, count_record=None, record_output=False):
        for i in range(len(trial_chronogram_early)):
            self.esn_output = self.esn.call({'reservoir_{}'.format(self.i_sim) + '_0': trial_chronogram_early[i].ravel(),
                                            'reservoir_{}'.format(self.i_sim) + '_1': trial_chronogram_late[i].ravel()})[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout[2].state()[0])
        self.select_choice()

    def train(self, reward, choice, trial):
        W_out = {}
        if sp.issparse(self.readout[2].params['Wout']):
            W_out[2] = np.array(self.readout[2].params['Wout'].todense())
        else:
            W_out[2] = np.array(self.readout[2].params['Wout'])
        if self.flag[2]:
            self.mask[2] = W_out[2] != 0
            self.flag[2] = False

        r = self.reservoir[2].state()
        W_out[2][:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r[0][:] - self.r_th))
        W_out[2] = W_out[2]*self.mask[2]
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[2][:, i])
            if col_norm != 0:
                W_out[2][:, i] = W_out[2][:, i]/col_norm
        self.readout[2].params['Wout'] = sp.csr_matrix(W_out[2])

        # print('trial', trial)
        # print('best choice', self.task.get_best_choice(trial))
        # print('cue', np.argmax(trial[:,self.task.get_best_choice(trial)]))

        # integ 2: readout 1 and 2 not trained
        #hyperopt integ 3
        """for i in range(2):
            if sp.issparse(self.readout[i].params['Wout']):
                W_out[i] =  np.array(self.readout[i].params['Wout'].todense())
            else:
                W_out[i] =  np.array(self.readout[i].params['Wout'])

            if self.flag[i]:
                self.mask[i] = W_out[i] != 0
                self.flag[i] = False

            r = self.reservoir[i].state()
            W_out[i] = np.array(W_out[i])

            cues = []
            for f in range(4):
                cues.append(self.readout[i].state()[0][choice + (4*f)])
            chosen_cue = np.argmax(cues)

            W_out[i][:, choice + (chosen_cue*4)] += np.array(
                self.eta * (reward - self.softmax(self.readout[i].state()[0])[choice + (4*chosen_cue)]) * (r[0][:] - self.r_th))
            W_out[i] = W_out[i] * self.mask[i]
            #self.all_W_out.append(W_out[i])
            for k in range(self.n_position):
                col_norm = np.linalg.norm(W_out[i][:, k])
                if col_norm != 0:
                    W_out[i][:, k] = W_out[i][:, k] / col_norm
            self.readout[i].params['Wout'] = sp.csr_matrix(W_out[i])""" # FAIL B F #Fail: nee

class Dual_separate_input_inhibition(Regular):
    def __init__(self, seed, filename, n_position=4, hyperparam_optim=False, lr=None, sr=None,
                  rc_connectivity=None, input_connectivity=None, fb_connectivity=None,
                 eta=None, beta=None, decay=None, i_sim=None,output_connectivity=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim,  lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity, fb_connectivity=fb_connectivity,
                   eta= eta, beta=beta, decay=decay, i_sim=i_sim, output_connectivity=output_connectivity)

        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.record_output_activity_testing = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = True
        print('Separate input:', self.separate_input)

    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              fb_connectivity=None, eta=None, beta=None, decay=None, i_sim=None,output_connectivity=None):

        _ = self.parameters
        self.i_sim =i_sim
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
        #self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.W = []
        self.fb_connectivity = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(2):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.fb_connectivity.append(fb_connectivity[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))

            #self.fb_connectivity.append(fb_connectivity[0])
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            for i in range(2):
                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i + 1))]['rc_connectivity'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i + 1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units),
                                     seed=self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), 2)

        def Wfb(*shape, seed=42, sr=None, **kwargs):
            return -np.random.binomial(size=(250, 250), n=1, p=0.5)

        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))
        self.reservoir = {}
        """for i in range(2):
            self.reservoir[i] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i], Wfb=Initializer(Wfb),
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
                                          activation='sigmoid',
                                          name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(i)))"""
        self.reservoir[0] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                      input_scaling=self.input_scaling, W=self.W[i],
                                      rc_connectivity=self.rc_connectivity[i],
                                      noise_rc=self.noise_rc,
                                      fb_scaling=self.fb_scaling,
                                      input_connectivity=self.input_connectivity[i], seed=seeds[i],
                                      activation='tanh',
                                      name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(0)))

        i = 1
        self.reservoir[1] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                      input_scaling=self.input_scaling, W=self.W[i], Wfb=Initializer(Wfb),
                                      rc_connectivity=self.rc_connectivity[i],
                                      noise_rc=self.noise_rc,
                                      fb_scaling=self.fb_scaling,
                                      input_connectivity=self.input_connectivity[i],
                                      fb_connectivity=self.fb_connectivity[1], seed=seeds[i],
                                      activation='tanh',
                                      name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(1)))

        self.reservoir[0] <<= self.reservoir[1]
        #self.reservoir[1] <<= self.reservoir[0]
        self.esn = [self.reservoir[0], self.reservoir[1]] >> self.readout
        np.random.seed(self.seed)
        random.seed(self.seed)

    def process(self, trial_chronogram_early, trial_chronogram_late, count_record=None, record_output=False, testing=False):
        for i in range(len(trial_chronogram_early)):
            self.esn_output = self.esn.call({'reservoir_{}'.format(self.i_sim) + '_0': trial_chronogram_early[i].ravel(),
                                            'reservoir_{}'.format(self.i_sim) + '_1': trial_chronogram_late[i].ravel()})[0]
            if record_output:
                if testing:
                    self.record_output_activity_testing[count_record]['output_activity'].append(self.readout.state()[0])
                else:
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


class Converging_separate_input(Regular):
    def __init__(self, seed, filename, n_position=4,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay= None, i_sim=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,eta=eta, beta=beta, decay=decay, i_sim=i_sim)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = True


    def setup(self, hyperparam_optim=False,  lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None, i_sim=None):

        _ = self.parameters
        self.i_sim = i_sim
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
        units = [225, 225, 50]
        for i in range(self.n_reservoir):
            self.units = units[i]
            self.reservoir[i] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=seeds[i],
                                          activation=activation_func,
                                          name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(i)))

        #self.reservoir[0] <<= self.readout
        #self.reservoir[2] <<= self.readout

        #self.esn = [self.reservoir[0], self.reservoir[1],
         #           [self.reservoir[0], self.reservoir[1]] >> self.reservoir[2]] >> self.readout

        self.esn = [[self.reservoir[0], self.reservoir[1]] >> self.reservoir[2]] >> self.readout


    def process(self, trial_chronogram_early, trial_chronogram_late, count_record=None, record_output=False):
        for i in range(len(trial_chronogram_early)):
            self.esn_output = self.esn.call({'reservoir_{}'.format(self.i_sim) + '_0': trial_chronogram_early[i].ravel(),
                                            'reservoir_{}'.format(self.i_sim) + '_1': trial_chronogram_late[i].ravel()})[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout.state()[0])
        self.select_choice()

    def train(self, reward, choice):
        if sp.issparse(self.readout.params['Wout']):
            W_out = np.array(self.readout.params['Wout'].todense())
        else:
            W_out = np.array(self.readout.params['Wout'])
        if self.flag:
            self.mask = W_out != 0
            self.flag = False
        r = self.reservoir[2].state()

        W_out[:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r[0][:] - self.r_th))
        W_out = W_out*self.mask
        self.all_W_out.append(W_out)

        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)

    def train_(self, reward, choice):
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


class BG_model(Regular):
    def __init__(self, seed, filename, n_position=4,
                 hyperparam_optim=False, n_units=None,lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None,decay=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, n_units=n_units,lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,eta=eta, beta=beta, decay=decay)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1


    def setup(self, hyperparam_optim=False,  n_units=None,lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay= None):

        _ = self.parameters
        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.output_connectivity = _['Readout']['output_connectivity']
        self.units = _['ESN_1']['n_units']
        self.noise_rc = _['ESN_1']['noise_rc']
        self.fb_scaling = _['ESN_1']['fb_scaling']
        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']
        self.lr = []
        self.sr = []
        self.n_units = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            for i in range(5):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                #self.n_units.append(n_units[i])
            for i in range(2):
                self.n_units.append(n_units[i])


        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            for i in range(5):
                self.lr.append(_['ESN_{}'.format(str(i+1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i+1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i+1))]['rc_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i+1))]['input_connectivity'])

        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), 5)
        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))

        self.reservoir  = {}
        for i, node in enumerate(('Str', 'GPi')):
            self.reservoir[node] = Reservoir(units=self.n_units[i], lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling,
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=seeds[i],
                                          activation=negative_sigmoid)

        for i, node in enumerate(('Ctx', 'STN', 'Th')):
            self.reservoir[node] = Reservoir(units=self.units, lr=self.lr[i+2], sr=self.sr[i+2],
                                             input_scaling=self.input_scaling,
                                             rc_connectivity=self.rc_connectivity[i+2],
                                             noise_rc=self.noise_rc,
                                             fb_scaling=self.fb_scaling,
                                             input_connectivity=self.input_connectivity[i+2],
                                             fb_connectivity=self.fb_connectivity, seed=seeds[i+2],
                                             activation='sigmoid')
        hyperdirect = self.reservoir['Ctx'] >> self.reservoir['STN']
        direct = self.reservoir['Ctx'] >> self.reservoir['Str']
        self.reservoir['Ctx'] <<= self.readout
        self.esn = [hyperdirect, direct] >> self.reservoir['GPi'] >> self.reservoir['Th'] >> self.readout


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

        W_out = np.array(W_out)

        W_out[:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (self.reservoir['Th'].state()[0][:] - self.r_th))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class Converging_functional(Regular):
    def __init__(self, seed, filename, n_position=4,
                 hyperparam_optim=False, n_units=None, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim, n_units=n_units, lr=lr, sr=sr, rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity, eta=eta, beta=beta, decay=decay)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1

    def setup(self, hyperparam_optim=False,  n_units=None, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None):

        _ = self.parameters
        self.n_reservoir = 3
        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.activation_func = _['ESN_1']['activation']
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
        self.n_units = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            for i in range(self.n_reservoir):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.n_units.append(n_units[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.n_units[i]),
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
                self.n_units.append(_['ESN_{}'.format(str(i+1))]['n_units'])


        def W_in(*shape, seed=42, sr=None, **kwargs):
            W_1 = -np.random.binomial(size=(self.n_units[2], self.n_units[0]), n=1, p=0.5)
            W_2 = np.random.binomial(size=(self.n_units[2], self.n_units[1]), n=1, p=0.5)
            return np.concatenate((W_1, W_2), axis=1)

        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), self.n_reservoir)
        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))
        self.reservoir  = {}
        for i in range(2):
            self.reservoir[i] = Reservoir(units=self.n_units[i], lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=self.seed,
                                          activation='sigmoid')
            self.reservoir[i] <<= self.readout

        self.reservoir[2] = Reservoir(units=self.n_units[2], lr=self.lr[2], sr=self.sr[2],
                                      input_scaling=self.input_scaling, Win=Initializer(W_in), W=self.W[2],
                                      rc_connectivity=self.rc_connectivity[2],
                                      noise_rc=self.noise_rc,
                                      fb_scaling=self.fb_scaling,
                                      input_connectivity=self.input_connectivity[2],
                                      fb_connectivity=self.fb_connectivity, seed=self.seed, activation='sigmoid')
        self.reservoir[i] <<= self.readout

        self.esn = [self.reservoir[0], self.reservoir[1],
                    [self.reservoir[0], self.reservoir[1]] >> self.reservoir[2]] >> self.readout

        #self.esn = [self.reservoir[0], self.reservoir[1]] >> self.reservoir[2] >> self.readout


    def train_(self, reward, choice):
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

        W_out = np.array(W_out)

        W_out[:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]) * (
                    self.reservoir[2].state()[0][:] - self.r_th))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)

    def train(self, reward, choice):
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
        beg = 0
        end = self.n_units[0]
        for i in range(self.n_reservoir):
            W_out_dict[i] = np.array(W_out[beg:end])
            W_out_dict[i][:, choice] += np.array(
                self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r_states[i][0][:] - self.r_th))
            if i < 2:
                beg += self.n_units[i]
                end += self.n_units[i+1]

        W_out = np.concatenate(tuple([W_out_dict[i] for i in range(self.n_reservoir)]))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i] / col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class Forward_readout(Regular):
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
        self.separate_input = False

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
        seeds = random.sample(range(1, 100), 2)

        self.readout = {}
        self.readout[0] = Ridge(16,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=seeds[0]))
        self.readout[1] = Ridge(self.n_position,
                                Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=seeds[1]))
        self.reservoir  = {}
        for i in range(2):
            self.reservoir[i] = Reservoir(units=self.units, lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity, seed=seeds[i],
                                          activation=activation_func)
            #self.reservoir[i] <<= self.readout[i]
            self.reservoir[i] <<= self.readout[1]
        part_0 = self.reservoir[0] >> self.readout[0]
        part_1 = self.reservoir[1] >> self.readout[1]
        self.esn = part_0 >> part_1

    def process(self, trial_chronogram, count_record=None, record_output= False):
        for i in range(len(trial_chronogram)):
            self.esn_output = self.esn.call(trial_chronogram[i].ravel())[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout[1].state()[0])
        self.select_choice()

    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        W_out = {}
        r = self.reservoir[1].state()
        if sp.issparse(self.readout[1].params['Wout']):
            W_out[1] = np.array(self.readout[1].params['Wout'].todense())
        else:
            W_out[1] = np.array(self.readout[1].params['Wout'])
        if self.flag:
            self.mask = W_out[1] != 0
            self.flag = False

        W_out[1][:, choice] += np.array(
            self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r[0][:] - self.r_th))
        W_out[1] = W_out[1] * self.mask
        self.all_W_out.append(W_out[1])

        for j in range(self.n_position):
            col_norm = np.linalg.norm(W_out[1][:, j])
            if col_norm != 0:
                W_out[1][:, j] = W_out[1][:, j] / col_norm
        self.readout[1].params['Wout'] = sp.csr_matrix(W_out[1])

        r = self.reservoir[0].state()
        if sp.issparse(self.readout[0].params['Wout']):
            W_out[0] = np.array(self.readout[0].params['Wout'].todense())
        else:
            W_out[0] = np.array(self.readout[0].params['Wout'])
        if self.flag:
            self.mask = W_out[0] != 0
            self.flag = False
        """softmax_sum = []
        for i in range(4):
            softmax_sum.append(self.softmax(self.readout[0].state()[0])[i + choice])
        softmax_mean = np.mean(softmax_sum)"""
        """max_softmax = 0
        for i in range(4):
            if self.softmax(self.readout[0].state()[0])[i + choice] > max_softmax:
                max_softmax = self.softmax(self.readout[0].state()[0])[i + choice]
        W_out[0][:, choice] += np.array(
            self.eta * (reward - max_softmax) * (r[0][:] - self.r_th))"""



















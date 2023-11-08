import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize  # Minimize function
import matplotlib.pyplot as plt


def filter_subjects(data, bias=0.8):
    """
    Filter valid subjects according to their bias between
    left and right that needs to be smaller than given bias.

    Parameters:
    -----------

    data : dataframe
      Database

    bias: float
      maximum bias

    Return:
    -------

    A tuple of valid ID and rejected ID lists
    """
    subjects_bias = {}
    subject_ids = data['subject_id'].unique()
    valid_ids, reject_ids = [], []
    for i, subject_id in enumerate(subject_ids):
        trials = select_trials(data, [subject_id])
        c0 = len(trials.loc[(trials['response'] == 0)]) / len(trials)
        c1 = len(trials.loc[(trials['response'] == 1)]) / len(trials)
        subjects_bias[subject_id] = c1 - 0.5

        if abs(c0 - c1) > bias:
            reject_ids.append(subject_id)
        else:
            valid_ids.append(subject_id)

    return valid_ids, reject_ids


def select_trials(data, subject_id=None, task_id=None):
    """
    Select all the trials for given individual(s) (subject_id) and task(s) (task_id).
    subject_id must be a subgroup of subject_ids, task_id mist be a subgroup of task_ids.

    Parameters:
    -----------

    data : dataframe
      Database

    subject_id: string or list
      subjects to be selected (ID)

    task_id: int or list
      tasks to be selected (ID)

    Return:
    -------

    A dataframe containing subject_id(s) and task_id(s)
    """

    if isinstance(subject_id, str):
        subject_id = [subject_id]

    if isinstance(task_id, int):
        task_id = [task_id]

    if subject_id is not None and task_id is not None:
        return data.loc[(data['task_id'].isin(task_id)) & (data['subject_id'].isin(subject_id))]
    elif subject_id is not None:
        return data.loc[(data['subject_id'].isin(subject_id))]
    elif task_id is not None:
        return data.loc[(data['task_id'].isin(task_id))]
    else:
        return data


def convert_trials(data, subject_id=None, task_id=None):
    """
    Convert trials from left/right to risky/safe for given individual(s) (subject_id)
    and task(s) (task_id). subject_id must be a subgroup of subject_ids, task_id must
    be a subgroup of task_ids.

    Parameters:
    -----------

    data : dataframe
      Database

    subject_id: string or list
      subjects to be selected (ID)

    task_id: int or list
      tasks to be selected (ID)

    Return:
    -------

    A converted and renamed dataframe (left/right replaced with risky/safe)
    """

    trials = select_trials(data, subject_id, task_id).copy()
    trials["bias"] = 0.0

    # We compute and store the left/right bias since it can be later used
    # for fitting and thus need to be transformed according to the risky/safe
    # paradigm. The bias is computed over all the tasks (i.e. not restricted
    # to the to b converted task_ids)
    for i, sid in enumerate(trials["subject_id"].unique()):
        T = select_trials(data, subject_id)

        # Right bias for task_id only
        B = len(T.loc[(T['response'] == 1)]) / len(T) - 0.5
        trials.loc[(trials['subject_id'] == sid), "bias"] = B

        # Right bias over all trials
        # trials.loc[(trials['subject_id'] == sid), "bias"] = subjects_bias[sid]

    P_left, V_left = trials['P_left'], trials['V_left']
    P_right, V_right = trials['P_right'], trials['V_right']

    I = P_right < P_left
    P_risky = np.where(I, P_right, P_left)
    V_risky = np.where(I, V_right, V_left)
    P_safe = np.where(I, P_left, P_right)
    V_safe = np.where(I, V_left, V_right)
    R = np.where(I, trials['response'], 1 - trials['response'])
    B = np.where(I, trials['bias'], -trials['bias'])

    trials = trials.rename(columns={"P_left": "P_risky",
                                    "V_left": "V_risky",
                                    "P_right": "P_safe",
                                    "V_right": "V_safe"})
    trials["P_risky"] = P_risky
    trials["V_risky"] = V_risky
    trials["P_safe"] = P_safe
    trials["V_safe"] = V_safe
    trials["response"] = R
    trials["bias"] = B

    return trials


def subjective_utility(X, p):
    """ Subjective utility of X """

    return np.where(X > 0, np.power(np.abs( X), p["rho"]),
                           -p["lambda"] * np.power(np.abs(X), p["rho"]))


def subjective_probability(X, p):
    "Subjective probability of X"

    return np.exp(- np.power((-np.log(X)), p["alpha"]))


def accept(X, bias, p=None):
    """
    Probability of accepting a gamble, given a difference X

    Inidividual biases are needed when X derives from the conversion
    of left/right to risky/safe. In other case, it can be a scalar.

    """

    if p is None:
        p = bias
        bias = p["bias"]

    P = 1 / (1.0 + np.exp(-p["mu"] * (X - p["x0"] + bias)))
    # P = 1/(1.0 + np.exp(-p["mu"]*(X - p["x0"])))
    epsilon = 1e-10
    return np.maximum(np.minimum(P, 1 - epsilon), epsilon)


def log_likelihood(Y, P_risky, V_risky, P_safe, V_safe, bias, p):
    """ Compute the log likelihood  """

    R = subjective_probability(P_risky, p) * subjective_utility(V_risky, p)
    S = subjective_probability(P_safe, p) * subjective_utility(V_safe, p)
    P = accept(R - S, bias, p)
    log_likelihood = Y * np.log(P) + (1 - Y) * (np.log(1 - P))
    return -log_likelihood.sum()


def evaluate(trials, p):
    """ Compate actual choices with their estimation """

    Y = trials["response"]
    P_risky, V_risky = trials["P_risky"], trials["V_risky"]
    P_safe, V_safe = trials["P_safe"], trials["V_safe"]
    bias = trials["bias"]

    R = subjective_probability(P_risky, p) * subjective_utility(V_risky, p)
    S = subjective_probability(P_safe, p) * subjective_utility(V_safe, p)
    P = accept(R - S, bias, p) > 0.5

    return Y.sum(), P.sum()


def objective(X, Y, P_risky, V_risky, P_safe, V_safe, bias):
    params = {"x0": X[0], "mu": X[1], "alpha": X[2], "lambda": X[3], "rho": X[4]}
    return log_likelihood(Y, P_risky, V_risky, P_safe, V_safe, bias, params)


def plot(ax, f, xlimits, fits, mean_fit):
    """ Convenience functon to factorize code"""

    xmin, xmax = xlimits
    X = np.linspace(xmin, xmax, 500)
    for p in fits.values():
        ax.plot(X, f(X, p), color="C0", lw=0.5, alpha=0.25)
    ax.plot(X, f(X, mean_fit), color="C0", lw=1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return X


def fit_procedure(valid_ids, data):
    skipped_trials = 50
    # Initial guess for parameters
    X0 = np.array([0,  # x0
                   1,  # mu
                   1,  # alpha
                   1,  # lambda
                   1])  # rho
    # Parameter bounds
    bounds = np.array([(-5, 5),  # x0
                       (0.01, 3),  # mu
                       (0.00, 2),  # alpha
                       (-2, 2),  # lambda
                       (0.0001, 1)])  # rho
    fits = {}
    with tqdm(total=len(valid_ids)) as progress:
        for subject_id in valid_ids:
            trials = select_trials(convert_trials(data, subject_id, [6])).copy()
            trials.sort_values(by='date', inplace=True)
            trials = trials[skipped_trials:]
            # Do we normalize values for keping things between -1, +1 ?
            Y = trials["response"]
            P_risky, V_risky = trials["P_risky"], trials["V_risky"]
            P_safe, V_safe = trials["P_safe"], trials["V_safe"]
            bias = trials["bias"]

            # X0 = np.random.uniform(low=bounds[:,0], high=bounds[:,1])
            # X0 = X0 + .75*np.random.uniform(-1,1,X0.shape)

            res = minimize(objective, x0=X0, bounds=bounds,
                           method="L-BFGS-B", tol=1e-10,
                           options={"maxiter": 1000, "disp": False},
                           args=(Y, P_risky, V_risky, P_safe, V_safe, bias))

            X = res.x
            params = {"x0": X[0], "mu": X[1], "alpha": X[2], "lambda": X[3], "rho": X[4]}
            fits[subject_id] = params
            progress.update()
    return fits


def evaluate(trials, p):
    """ Compate actual choices with their estimation """

    Y = trials["response"]
    P_risky, V_risky = trials["P_risky"], trials["V_risky"]
    P_safe, V_safe = trials["P_safe"], trials["V_safe"]
    bias = trials["bias"]

    R = subjective_probability(P_risky, p) * subjective_utility(V_risky, p)
    S = subjective_probability(P_safe, p) * subjective_utility(V_safe, p)
    P = accept(R - S, bias, p) > 0.5

    return Y.sum(), P.sum()


def evaluate_model(valid_ids, fits, data):
    mean_perf = 0
    skipped_trials = 50
    for subject_id in valid_ids:
        params = fits[subject_id]
        trials = convert_trials(select_trials(data, subject_id, [6])).copy()
        trials.sort_values(by='date', inplace=True)
        trials = trials[skipped_trials // 2:]
        truth, model = evaluate(trials, params)
        print("%s: Gain (n=%5d): %.2f (%d / %d)" % (subject_id, len(trials), truth / max(model, 1), truth, model))
        mean_perf += .5 * abs(1 - truth / max(model, 1))

        # trials = convert_trials(select_trials(data, subject_id, [7])).copy()
        # trials.sort_values(by='date', inplace = True)
        # trials = trials[skipped_trials//2:]

        # truth, model = evaluate(trials, params)
        # print("     Loss (n=%5d): %.2f (%d / %d)" % (len(trials), truth/max(model,1), truth, model))
        # mean_perf += .5*abs(1-truth/max(model,1))

        # print()

    print("Mean accuracy: %.1f %%" % (100 - (100 * mean_perf / len(valid_ids))))


def compute_mean_fit(data, subject_id, fits):
    # Mean parameters
    mean_fit = {}
    params = fits[subject_id]
    for pname in params.keys():
        mean_fit[pname] = np.mean([fits[sid][pname] for sid in fits.keys()])

    # Mean bias (we could have stored biases when filtering valid subjects)
    mean_bias = 0
    for sid in fits.keys():
        T = select_trials(data, subject_id, [6, 7])
        # Right bias
        bias = len(T.loc[(T['response'] == 1)]) / len(T) - 0.5
        fits[sid]["bias"] = bias
        mean_bias += bias

    mean_fit["bias"] = mean_bias / len(fits)
    return mean_fit


def plot_prospect_fit(mean_fit, fits):
    #fig = plt.figure(figsize=(10, 10), dpi=200)
    # Subjective probability
    # ----------------------
    ax = plt.subplot(1, 3, 1, aspect=1)
    X = plot(ax, subjective_probability, xlimits=(0.01, 1), fits=fits, mean_fit=mean_fit)
    ax.plot(X, X, color="black", lw=0.5, ls="--")

    x0 = np.exp(-1)
    ax.axvline(x0, color="black", ls="--", lw=.75)
    ax.axhline(x0, color="black", ls="--", lw=.75)
    ax.set_title("w(p)")
    ax.set_xticks([0, x0, 1])
    ax.set_xticklabels(["0", "1/e", 1])
    ax.set_yticks([0, x0, 1])
    ax.set_yticklabels(["0", "1/e", 1])
    ax.text(1, 0.025, r"$\alpha = %.2f$" % mean_fit["alpha"],
            ha="right", alpha=.5, transform=ax.transAxes)

    # Subjective utility
    # ------------------
    ax = plt.subplot(1, 3, 2, aspect=.72)
    X = plot(ax, subjective_utility, xlimits=(-1, 1),fits=fits, mean_fit=mean_fit)
    ax.plot(X, X, color="black", lw=0.5, ls="--")

    ax.axvline(0, color="black", ls="--", lw=.75)
    ax.axhline(0, color="black", ls="--", lw=.75)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_title("u(x)")
    ax.text(1, 0.025, r"$\lambda = %.2f, \rho=%.2f$" % (mean_fit["lambda"], mean_fit["rho"]),
            ha="right", alpha=.5, transform=ax.transAxes)

    # Accept
    # ------
    ax = plt.subplot(1, 3, 3, aspect=4.1)
    X = plot(ax, accept, xlimits=(-2, 2),fits=fits, mean_fit=mean_fit)
    ax.plot(X, accept(X, {"x0": 0, "mu": 1, "bias": 0}), color="black", lw=0.5, ls="--")
    ax.axvline(0.0, color="black", ls="--", lw=.75)
    ax.axhline(0.5, color="black", ls="--", lw=.75)
    ax.set_title("accept(Δx)")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([0, 0.5, 1])
    ax.text(1, 0.025, r"$\mu = %.2f, x_0=%.2f$" % (mean_fit["mu"], mean_fit["x0"]),
            ha="right", alpha=.5, transform=ax.transAxes);

    plt.tight_layout()
    plt.show()


def run_prospect_fit(df):
    #valid_ids, reject_ids = filter_subjects(df)
    subject_ids = df['subject_id'].unique()
    task_ids = list(range(1, 8))
    fits = fit_procedure(subject_ids, df)
    print(fits)
    mean_fit = compute_mean_fit(df, 'esn', fits)
    evaluate_model(subject_ids,fits, df)
    plot_prospect_fit(mean_fit, fits)




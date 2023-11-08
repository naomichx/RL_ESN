import numpy as np
import matplotlib.pyplot as plt
import statistics as st
cmap = plt.get_cmap("tab20b", 20)

n_unseen = [5, 10, 20, 30, 40, 50, 60, 70]
model_types = ('regular', 'parallel', 'forward',  'dual','differential', 'converging',)#,differential
model_types = ('differential', 'regular')
deterministic = False

if deterministic:
    path = "results/same_parameters/deterministic/"
else:
    path = 'results/same_parameters/stochastic/'
path += 'val_gen_test/'

labels = []
paths = []
for model_type in model_types:
    paths.append(path + model_type + '/')
    labels.append(model_type)


C = [cmap(2 + i * 4) for i in range(len(paths))]
k = 0
for i, path in enumerate(paths):
    res = np.load(path + model_types[i] + '_gen_test_result.npy', allow_pickle=True).item()
    print(res)
    x = []
    y = []
    y_lower = []
    y_upper = []
    for n in n_unseen:
        x.append(n)
        y.append(res['n_unseen {}'.format(n)]['mean'])
        y_lower.append(res['n_unseen {}'.format(n)]['mean'] - res['n_unseen {}'.format(n)]['std'])
        y_upper.append(res['n_unseen {}'.format(n)]['mean'] + res['n_unseen {}'.format(n)]['std'])
    plt.plot(x, y, ls='-', color=C[k], label=model_types[k])
    plt.fill_between(x, y_lower, y_upper, facecolor=C[k], alpha=0.15)
    k += 1

plt.legend()
plt.ylim((10, 100))
plt.title('Value generalization test for the Stochastis task')
plt.xlabel('Number of unseen combinations')
plt.ylabel('% of successful decisions averaged on 10 trials ')
plt.tight_layout()
plt.show()
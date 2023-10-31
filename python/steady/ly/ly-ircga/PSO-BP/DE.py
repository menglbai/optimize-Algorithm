'''
min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
s.t.
    x1*x2 >= 1
    x1*x2 <= 5
    x2 + x3 = 1
    0 <= x1, x2, x3 <= 5
'''

from sko.DE import DE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def obj_func(p):
    x1, x2, x3 = p
    return x1 ** 2 + x2 ** 2 + x3 ** 2


# constraint_eq = [
#     lambda x: 1 - x[1] - x[2]
# ]
#
# constraint_ueq = [
#     lambda x: 1 - x[0] * x[1],
#     lambda x: x[0] * x[1] - 5
# ]


de = DE(func=obj_func, n_dim=3, size_pop=20, max_iter=50, lb=[0, 0, 0], ub=[5, 5, 5])
best_x, best_y = de.run()

print('best_x:', best_x, '\n', 'best_y:', best_y)
print(de.generation_best_Y)
print(de.all_history_Y)
Y_history = pd.DataFrame(de.all_history_Y)
gbest_history = np.array(Y_history.min(axis=1).cummin())
print(gbest_history)
plt.plot(gbest_history)
plt.show()

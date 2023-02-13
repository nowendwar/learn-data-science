from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import time
start_time = time.time()

problem = get_problem("zdt1")

print('problem', problem)

algorithm = NSGA2(pop_size=100)

print('algorithm', algorithm)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=False)

print('res', res)
print('res.message', res.message)

# print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

# plot = Scatter()
# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
# plot.add(res.F, facecolor="none", edgecolor="red")
# plot.show()


print("--- %s seconds ---" % (time.time() - start_time))
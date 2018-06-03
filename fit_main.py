import numpy as np 
import nlopt

# We'll optimize Himmelblau's function

def opt_me(x, grad):
    a, b = x[0], x[1]
    return (a**2 + b - 11)**2 + (a + b**2 - 7)**2 

# Optimization stage

maxeval = 100
minrms = 0.01
tol = 0.0001

param_values = np.array([0, 0])

opt = nlopt.opt(nlopt.G_MLSL_LDS, 2)
opt.set_local_optimizer(nlopt.opt(nlopt.LN_BOBYQA, 2))
opt.set_lower_bounds(np.array([-5, -5]))
opt.set_upper_bounds(np.array([5, 5]))
opt.set_min_objective(opt_me)
opt.set_maxeval(maxeval)
opt.set_stopval(minrms) 
opt.set_ftol_abs(tol)
x = opt.optimize(param_values)
minf = opt.last_optimum_value()

with open('out', 'w') as outfile:
     outfile.write("optimum at " + str(x[0]) + " " + str(x[1]) + "\n")
     outfile.write("minimum value = " + str(minf))



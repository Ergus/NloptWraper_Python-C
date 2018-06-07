import numpy as np 
import ctypes as C
#import nlopt

# Load shared library 
libfit = C.CDLL("./libnlopt.so.0.9.0")

# We'll optimize Himmelblau's function

def opt_me(x, grad):
    a, b = x[0], x[1]
    return (a**2 + b - 11)**2 + (a + b**2 - 7)**2

################################################################
# NLopt SWIG-python wrapper
################################################################

#maxeval = 100
#minrms = 0.01
#tol = 0.0001
param_values = np.array([0, 0], dtype=np.float64)

#opt = nlopt.opt(nlopt.G_MLSL_LDS, 2)
#opt.set_local_optimizer(nlopt.opt(nlopt.LN_BOBYQA, 2))
#opt.set_lower_bounds(np.array([-5, -5]))
#opt.set_upper_bounds(np.array([5, 5]))
#opt.set_min_objective(opt_me)
#opt.set_maxeval(maxeval)
#opt.set_stopval(minrms) 
#opt.set_ftol_abs(tol)
#x = opt.optimize(param_values)
#minf = opt.last_optimum_value()

################################################################
# CTypes wrapper
################################################################

# Objective function ported to C 

CMPFUNC = C.CFUNCTYPE(C.c_double, np.ctypeslib.ndpointer(C.c_double, flags="C_CONTIGUOUS"))
cmp_opt_me = CMPFUNC(opt_me)

# Parameters ported to C types

maxeval = C.c_int(100)
minrms = C.c_double(0.01)
tol = C.c_double(0.0001)
minf = C.c_double()

# Fucntion prototyping

libfit.nlopt_create.argtypes = [C.c_int, C.c_int]
libfit.nlopt_create.restype = C.c_void_p

libfit.nlopt_set_lower_bounds.argtypes = [C.c_void_p ,
       np.ctypeslib.ndpointer(C.c_double, flags="C_CONTIGUOUS")]
libfit.nlopt_set_upper_bounds.argtypes = [C.c_void_p ,
       np.ctypeslib.ndpointer(C.c_double, flags="C_CONTIGUOUS")]

libfit.nlopt_set_min_objective.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p ]
libfit.nlopt_set_maxeval.argtypes = [C.c_void_p, C.c_int]
libfit.nlopt_set_stopval.argtypes = [C.c_void_p, C.c_double]
libfit.nlopt_set_ftol_abs.argtypes = [C.c_void_p, C.c_double]

libfit.nlopt_optimize.argtypes = [C.c_void_p, np.ctypeslib.ndpointer(C.c_double, flags="C_CONTIGUOUS"),
                                 C.c_double]
#libfit.nlopt_optimize.restypes = [np.ctypeslib.ndpointer(C.c_double, flags="C_CONTIGUOUS"), C.c_double]

# Optimization

#opt = libfit.nlopt_create('NLOPT_G_MLSL_LDS', 2)
#libfit.nlopt_set_local_optimizer(opt, libfit.nlopt_create('NLOPT_LN_BOBYQA', 2))
opt = libfit.nlopt_create(C.c_int(27), C.c_int(2))
libfit.nlopt_set_lower_bounds(opt, np.array([-5, -5], dtype=np.float64))
libfit.nlopt_set_upper_bounds(opt ,np.array([5, 5], dtype=np.float64))
libfit.nlopt_set_min_objective(opt, cmp_opt_me, None)
libfit.nlopt_set_maxeval(opt, maxeval)
libfit.nlopt_set_stopval(opt, minrms)
libfit.nlopt_set_ftol_abs(opt, tol)
libfit.nlopt_optimize(opt, param_values, minf)
libfit.nlopt_destroy(opt)

#with open('out', 'w') as outfile:
#     outfile.write("optimum at " + str(x[0]) + " " + str(x[1]) + "\n")
#     outfile.write("minimum value = " + str(minf))



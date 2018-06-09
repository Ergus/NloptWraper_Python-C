
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include "nlopt.h"

// The callback will be in a global variable
// This can be improved
static PyObject *callback;


// CountDict type
typedef struct {
    PyObject_HEAD
	int algorithm;
	unsigned n;
	nlopt_opt opt;
} Nlopt;

// Function that executes the callback
double exec_callback(unsigned n, const double *x,
                     double *grad, void *func_data)
{
	npy_intp dims[] = {n};

	PyObject *Ox = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void *) x);
	PyObject *Ograd = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, grad);

	if (!Ox || !Ograd)
		abort();

	PyObject *arglist = Py_BuildValue("OO", Ox, Ograd);

	PyObject *result = PyEval_CallObject(callback, arglist);

	Py_DECREF(arglist);
	Py_DECREF(Ograd);
    Py_DECREF(Ox);

    const double ret = PyFloat_AsDouble(result);

    return ret;
}

// This sets the global callback object.
static PyObject *Nlopt_set_callback(Nlopt *self, PyObject *args)
{
	PyObject *result = NULL, *temp = NULL;
	nlopt_result out;

	if (PyArg_ParseTuple(args, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        Py_XINCREF(temp);         /* Add a reference to new callback */
        Py_XDECREF(callback);  /* Dispose of previous callback */
        callback = temp;       /* Remember new callback */
        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;

        out = nlopt_set_min_objective(self->opt, exec_callback, NULL);

        if (out != NLOPT_SUCCESS)
	        return NULL;
	}
    return result;
}

// Object allocation in memory (no initialize)
static PyObject *Nlopt_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	Nlopt *self = (Nlopt *) type->tp_alloc(type, 0);
    self->algorithm = -1;
    self->n = 0;
    self->opt = NULL;
    return (PyObject *) self;
}

// Object initialization. (this receives parameters)
static int Nlopt_init(Nlopt *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = {"algorithm", "n", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iI", kwlist,
	                                 &self->algorithm, &self->n))
		return -1;

	if (self->algorithm >= NLOPT_NUM_ALGORITHMS)
		return -2;

	self->opt = nlopt_create(self->algorithm, self->n);

	if (!self->opt)
		return -3;

    return 0;
}

// Destructor
static void Nlopt_dealloc(Nlopt *self)
{
	nlopt_destroy(self->opt);
	Py_TYPE(self)->tp_free((PyObject *) self);
}

// Here start the Wrappers.
static PyObject *Nlopt_set_lower_bounds(Nlopt *self, PyObject *args)
{
	PyObject *ptr;

	if (!PyArg_ParseTuple(args, "O", &ptr))
		return NULL;

	PyArrayObject *array = (PyArrayObject *) ptr;

	const double *data = (double *) PyArray_DATA(array);

	nlopt_result out = nlopt_set_lower_bounds(self->opt, data);

	if (out != NLOPT_SUCCESS)
		return NULL;

	return Py_BuildValue("i", out);
}


static PyObject *Nlopt_set_upper_bounds(Nlopt *self, PyObject *args)
{
	PyObject *ptr;

	if (!PyArg_ParseTuple(args, "O", &ptr))
		return NULL;

	PyArrayObject *array = (PyArrayObject *) ptr;

	const double *data = (double *) PyArray_DATA(array);

	nlopt_result out = nlopt_set_upper_bounds(self->opt, data);

	if (out != NLOPT_SUCCESS)
		return NULL;

	return Py_BuildValue("i", out);
}


static PyObject *Nlopt_set_maxeval(Nlopt *self, PyObject *args)
{
	int maxeval;

	if (!PyArg_ParseTuple(args, "i", &maxeval))
		return NULL;

	nlopt_result out = nlopt_set_maxeval(self->opt, maxeval);

	if (out != NLOPT_SUCCESS)
		return NULL;

	return Py_BuildValue("i", out);
}


static PyObject *Nlopt_set_stopval(Nlopt *self, PyObject *args)
{
	double stopval;

	if (!PyArg_ParseTuple(args, "d", &stopval))
		return NULL;

	nlopt_result out = nlopt_set_stopval(self->opt, stopval);

	if (out != NLOPT_SUCCESS)
		return NULL;

	return Py_BuildValue("i", out);
}


static PyObject *Nlopt_set_ftol_abs(Nlopt *self, PyObject *args)
{
	double tol;

	if (!PyArg_ParseTuple(args, "d", &tol))
		return NULL;

	nlopt_result out = nlopt_set_ftol_abs(self->opt, tol);

	if (out != NLOPT_SUCCESS)
		return NULL;

	return Py_BuildValue("i", out);
}


static PyObject *Nlopt_optimize(Nlopt *self, PyObject *args)
{
	PyObject *Px, *Popt_f;

	if (!PyArg_ParseTuple(args, "OO", &Px, &Popt_f))
		return NULL;

	PyArrayObject *x = (PyArrayObject *) Px;
	PyArrayObject *opt_f = (PyArrayObject *) Popt_f;

	double *dx = (double *) PyArray_DATA(x);
	double *dopt_f = (double *) PyArray_DATA(opt_f);

	nlopt_result out = nlopt_optimize(self->opt, dx, dopt_f);

	if (out != NLOPT_SUCCESS)
		return NULL;

	return Py_BuildValue("i", out);
}

// ====== Defining the type. This parte exposes the object to python =======

// Object Members
static PyMemberDef Nlopt_members[] = {
	{"algorithm",  T_INT, offsetof(Nlopt, algorithm), 0, "The number of the algorithm."},
	{"n",  T_UINT, offsetof(Nlopt, n), 0, "The number n."},
	{NULL}
};

// Object functions
static PyMethodDef Nlopt_methods[] = {
	{"set_lower_bounds", (PyCFunction) Nlopt_set_lower_bounds,
	 METH_VARARGS, "Set lower bounds."},
	{"set_upper_bounds", (PyCFunction) Nlopt_set_upper_bounds,
	 METH_VARARGS, "Set upper bounds."},
	{"set_maxeval", (PyCFunction) Nlopt_set_maxeval,
	 METH_VARARGS, "Set maxeval."},
	{"set_stopval", (PyCFunction) Nlopt_set_stopval,
	 METH_VARARGS, "Set stopval."},
	{"set_ftol_abs", (PyCFunction) Nlopt_set_ftol_abs,
	 METH_VARARGS, "Set ftol abs."},
	{"optimize", (PyCFunction) Nlopt_optimize,
	 METH_VARARGS, "Execute Optimization."},
	{"set_callback", (PyCFunction) Nlopt_set_callback,
	 METH_VARARGS, "Sets the callback for the min_objective."},
    {NULL}
};

// This is the python object design to put all together.
// The PyObjectType  it an object itself
static PyTypeObject PyNloptType = {
	PyObject_HEAD_INIT(NULL)
    .tp_name = "PyNlopt",
	.tp_basicsize = sizeof(Nlopt),
	.tp_dealloc = (destructor)Nlopt_dealloc,
    .tp_doc = "PyNlopt Object",
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
	.tp_new = Nlopt_new,
	.tp_methods = Nlopt_methods,
	.tp_members = Nlopt_members,
	.tp_init = (initproc)Nlopt_init
};

// Defining the module
static PyModuleDef nloptmodule = {
	PyModuleDef_HEAD_INIT,
    .m_name = "wnlopt",
    .m_doc = "C wraper for nlopt.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_wnlopt(void)
{
	// Tests the object type 
    if (PyType_Ready(&PyNloptType) < 0)
        return NULL;

    // Creates the module object
    PyObject *m = PyModule_Create(&nloptmodule);
    if (!m)
	    return NULL;

    import_array();             // needed to use numpy

    Py_INCREF(&PyNloptType);

    // Add the python object to this module.
	PyModule_AddObject(m, "PyNlopt" , (PyObject* )&PyNloptType);

    return m;
}

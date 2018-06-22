/*
 * This file is part of the NloptWraper_Python-C distribution Copyright (c) 2017
 * Jimmy Aguilar Mena.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>
#include <Python.h>

#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"   // numpy objects

#include "nlopt.h"
#include "nlopt-enum.h"

// CountDict type
typedef struct {
    PyObject_HEAD
	nlopt_opt opt;
	PyObject *callback;
} Nlopt;


// Exception handling
static PyObject *_checkNlopt(int out, const char file[],
                             int line, const char function[])
{
	if (out != NLOPT_SUCCESS) {
		PyErr_Format(PyExc_RuntimeError,
		             "%s:%d %s -> Nlopt C function returned: %d expected: %d\n",
		             file, line, function, out, NLOPT_SUCCESS);
		return NULL;
	}
	return Py_BuildValue("i", out);
}

#define checkNlopt(out) _checkNlopt(out, __FILE__, __LINE__, __FUNCTION__);


// Object allocation in memory (no initialize)
static PyObject *Nlopt_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	Nlopt *self = (Nlopt *) type->tp_alloc(type, 0);
    self->opt = NULL;
    return (PyObject *) self;
}

// Object initialization. (this receives parameters)
static int Nlopt_init(Nlopt *self, PyObject *args, PyObject *kwds)
{
	int alg;
	unsigned n;
	static char *kwlist[] = {"algorithm", "n", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iI", kwlist,  &alg, &n))
		return -1;

	if (alg >= NLOPT_NUM_ALGORITHMS)
		return -2;

	self->opt = nlopt_create(alg, n);

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
static PyObject *Nlopt_set_lower_bounds(Nlopt *self, PyObject *arg)
{
	#ifndef NDEBUG
	if (!PyArray_Check(arg)) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> variable not an array",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}
	#endif

	PyArrayObject *array = (PyArrayObject *) arg;
	if (!array) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> Is not an array",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}

	const double *data = (double *) PyArray_DATA(array);
	const nlopt_result out = nlopt_set_lower_bounds(self->opt, data);
	return checkNlopt(out);
}


static PyObject *Nlopt_set_upper_bounds(Nlopt *self, PyObject *arg)
{
	#ifndef NDEBUG
	if (!PyArray_Check(arg)) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> variable not an array",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}
	#endif

	PyArrayObject *array = (PyArrayObject *) arg;
	if (!array) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> Is not an array",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}

	const double *data = (double *) PyArray_DATA(array);
	const nlopt_result out = nlopt_set_upper_bounds(self->opt, data);
	return checkNlopt(out);
}


static PyObject *Nlopt_set_maxeval(Nlopt *self, PyObject *arg)
{
	#ifndef NDEBUG
	if (!PyLong_Check(arg)) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> arg not an int",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}
	#endif

	const int maxeval = PyLong_AsLong(arg);
	const nlopt_result out = nlopt_set_maxeval(self->opt, maxeval);
	return checkNlopt(out);
}


static PyObject *Nlopt_set_stopval(Nlopt *self, PyObject *arg)
{
	#ifndef NDEBUG
	if (!PyFloat_Check(arg)) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> arg not a float",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}
	#endif

	const double stopval = PyFloat_AsDouble(arg);
	const nlopt_result out = nlopt_set_stopval(self->opt, stopval);
	return checkNlopt(out);
}


static PyObject *Nlopt_set_ftol_abs(Nlopt *self, PyObject *arg)
{
	#ifndef NDEBUG
	if (!PyFloat_Check(arg)) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> arg not a float",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}
	#endif

	const double tol = PyFloat_AsDouble(arg);
	const nlopt_result out = nlopt_set_ftol_abs(self->opt, tol);
	return checkNlopt(out);
}


static PyObject *Nlopt_optimize(Nlopt *self, PyObject *args, PyObject *kwds)
{
	PyObject *Px;
	double opt_f;
	char *kwlist[] = {"x", "opt_f", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "Of", kwlist,
	                                 &Px, &opt_f))
		return NULL;

	#ifndef NDEBUG
	if (!PyArray_Check(Px)) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> variable not an array",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}
	#endif

	PyArrayObject *x = (PyArrayObject *) Px;
	double *dx = (double *) PyArray_DATA(x);

	#ifndef NDEBUG
	const int n = nlopt_get_dimension(self->opt);
	printf("Algorithm: %d\n", nlopt_get_algorithm(self->opt));
	printf("Dimensions: %u\n", n);
	double upper[n], lower[n];
	nlopt_get_upper_bounds(self->opt, upper);
	nlopt_get_lower_bounds(self->opt, lower);
	printf("Upper [%lf; %lf]\n", upper[0], upper[1]);
	printf("Lower [%lf; %lf]\n", lower[0], lower[1]);
	printf("Maxeval %d\n", nlopt_get_maxeval(self->opt));
	printf("Stopval %lf\n", nlopt_get_stopval(self->opt));
	printf("Ftol_abs %lf\n", nlopt_get_ftol_abs(self->opt));
	#endif

	const nlopt_result out = nlopt_optimize(self->opt, dx, &opt_f);

	if (out < 0) {
		PyErr_Format(PyExc_RuntimeError,
		             "%s:%d %s -> Nlopt C function returned: %d expected: %d\n",
		             __FILE__, __LINE__, __FUNCTION__, out, NLOPT_SUCCESS);
		return NULL;
	}

	return Py_BuildValue("f", opt_f);
}


static PyObject *Nlopt_set_local_optimizer(Nlopt *self, PyObject *arg)
{
	if (!arg) {
		PyErr_Format(PyExc_RuntimeError,
		             "%s:%d %s -> Input opt is null\n",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}

	Nlopt *local_opt = (Nlopt *) arg;

	const nlopt_result out = nlopt_set_local_optimizer(self->opt, local_opt->opt);
	return checkNlopt(out);
}


// Functions for the callback
double callback(unsigned n, const double *x, double *grad, void *func_data)
{
	assert(func_data && n > 0);

	npy_intp dims = n;

	PyObject *Ox = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, (void *) x);
	PyObject *Ograd = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, grad);

	if (!Ox || !Ograd) {
		PyErr_Format(PyExc_RuntimeError,
		             "%s:%d %s -> could not make PyArray from C array\n",
		             __FILE__, __LINE__, __FUNCTION__);
		abort();
	}

	PyObject *result = PyObject_CallFunction(func_data, "OO", Ox, Ograd);

	if (!result) {
		PyErr_Format(PyExc_RuntimeError, "%s:%d %s -> result is null\n",
		             __FILE__, __LINE__, __FUNCTION__);
		abort();
	}

	const double ret = PyFloat_AsDouble(result);

    #ifndef NDEBUG
	printf("f(%lf, %lf) = %lf\n", x[0], x[1], ret);
	#endif

    return ret;
}

// This sets the global callback object.
static PyObject *Nlopt_set_callback(Nlopt *self, PyObject *arg)
{
	if (!PyCallable_Check(arg)) {
		PyErr_Format(PyExc_TypeError, "%s:%d %s -> Needed callable parameter",
		             __FILE__, __LINE__, __FUNCTION__);
		return NULL;
	}

	// Boilerplate to return "None"
	const nlopt_result out = nlopt_set_min_objective(self->opt, callback, arg);

	return checkNlopt(out);
}

// ====== Defining the type. This parte exposes the object to python =======

// Object Members
static PyMemberDef Nlopt_members[] = {
	{NULL}
};

// Object functions
static PyMethodDef Nlopt_methods[] = {
	{"set_lower_bounds", (PyCFunction) Nlopt_set_lower_bounds,
	 METH_O, "Set lower bounds."},
	{"set_upper_bounds", (PyCFunction) Nlopt_set_upper_bounds,
	 METH_O, "Set upper bounds."},
	{"set_local_optimizer", (PyCFunction) Nlopt_set_local_optimizer,
	 METH_O, "Set local optimizer."},
	{"set_maxeval", (PyCFunction) Nlopt_set_maxeval,
	 METH_O, "Set maxeval."},
	{"set_stopval", (PyCFunction) Nlopt_set_stopval,
	 METH_O, "Set stopval."},
	{"set_ftol_abs", (PyCFunction) Nlopt_set_ftol_abs,
	 METH_O, "Set ftol abs."},
	{"optimize", (PyCFunction) Nlopt_optimize,
	 METH_KEYWORDS | METH_VARARGS, "Execute Optimization."},
	{"set_callback", (PyCFunction) Nlopt_set_callback,
	 METH_O, "Sets the callback for the min_objective."},
	{NULL, NULL, 0, NULL}        /* Sentinel */
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
	#ifndef NDEBUG
	fprintf(stderr, "PYNLOPT using debug mode\n");
	#endif

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

	ADDVALUES(m);  // Adds the variables to the 

	return m;
}

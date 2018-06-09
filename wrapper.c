
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include "nlopt.h"

// CountDict type
typedef struct {
    PyObject_HEAD
	int algorithm;
	unsigned n;
    nlopt_opt opt;
} Nlopt;

static PyObject *Nlopt_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	Nlopt *self = (Nlopt *) type->tp_alloc(type, 0);
    self->algorithm = -1;
    self->n = 0;
    self->opt = NULL;
    return (PyObject *) self;
}

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


static void Nlopt_dealloc(Nlopt *self)
{
	nlopt_destroy(self->opt);
	Py_TYPE(self)->tp_free((PyObject *) self);
}


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

//NLOPT_EXTERN(nlopt_result) nlopt_set_min_objective(nlopt_opt opt,
//                                                   nlopt_func f,
//                                                   void *f_data);

// Defining the type

static PyMemberDef Nlopt_members[] = {
	{"algorithm",  T_INT, offsetof(Nlopt, algorithm), 0, "The number of the algorithm."},
	{"n",  T_UINT, offsetof(Nlopt, n), 0, "The number n."},
	{NULL}
};

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
    {NULL}
};

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
    if (PyType_Ready(&PyNloptType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&nloptmodule);
    if (!m)
	    return NULL;

    Py_INCREF(&PyNloptType);
	PyModule_AddObject(m, "PyNlopt" , (PyObject* )&PyNloptType);

    return m;
}

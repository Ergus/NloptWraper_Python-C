
#include "nlopt.h"
#include <stdio.h>
#include <stdlib.h>


double opt_me(int n, const double *x, double *grad, void *func_data)
{
	const double a = x[0];
	const double b = x[1];
	const double v1 = (a*a + b - 11);
	const double v2 = (a + b*b - 7);

	double ret = v1*v1 + v2*v2;

	#ifndef NDEBUG
	printf("f(%lf, %lf) = %lf\n", a, b, ret);
	#endif

	return ret;
}


int main() {

	const int n = 2;
	int maxeval = 1000;
	double minrms = 0.01;
	double tol = 0.0001;

	double lower[] = {-5., -5.};
	double upper[] = {5., 5.};
	double param_values[] = {0., 0.};
	double minf = 0.0;

	printf("%d %d\n", NLOPT_G_MLSL_LDS, NLOPT_LN_BOBYQA);

	nlopt_opt opt = nlopt_create(NLOPT_G_MLSL_LDS, n);
	nlopt_opt lopt = nlopt_create(NLOPT_LN_BOBYQA, n);
	nlopt_set_local_optimizer(opt, lopt);

	nlopt_set_lower_bounds(opt, lower);
	nlopt_set_upper_bounds(opt, upper);

	nlopt_set_min_objective(opt, opt_me, NULL);
	nlopt_set_maxeval(opt, maxeval);
	nlopt_set_stopval(opt, minrms);
	nlopt_set_ftol_abs(opt, tol);

	#ifndef NDEBUG
	printf("Algorithm: %d\n", nlopt_get_algorithm(opt));
	double lupper[n], llower[n];
	nlopt_get_upper_bounds(opt, lupper);
	nlopt_get_lower_bounds(opt, llower);
	printf("Upper [%lf; %lf]\n", upper[0], upper[1]);
	printf("Lower [%lf; %lf]\n", lower[0], lower[1]);
	printf("Maxeval %d\n", nlopt_get_maxeval(opt));
	printf("Stopval %lf\n", nlopt_get_stopval(opt));
	printf("Ftol_abs %lf\n", nlopt_get_ftol_abs(opt));
	#endif

	int dbg = nlopt_optimize(opt, param_values, &minf);

	if (dbg < 0) {
		fprintf(stderr, "%s:%d %s -> Nlopt C function failed: %d expected: %d\n",
		        __FILE__, __LINE__, __FUNCTION__, dbg, NLOPT_SUCCESS);
	} else {
		printf("minimum: f(%lf, %lf) = %lf\n",
		       param_values[0], param_values[1], minf);
	}

	nlopt_destroy(opt);

	return 0;
}


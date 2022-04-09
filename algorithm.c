#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void gradient_descent_with_k();
void gradient_descent_with_momentum();
double valueonly_beale2d(int dim, double *x);
double valueandderivatives_beale2d(int dim, double *x , double* grad, double *hessian_vecshaped);


int main() {
    // gradient_descent_with_k();
    gradient_descent_with_momentum();
    return 0;
}


void gradient_descent_with_k() {
    /*
        Optimzation Loop is breakable only:
        1. Boundary field of domain range has been reached / breached (Not a choice)
        2. Desired Threshold for intervals between iteration is satisfied (A choice)
        3. Desired Iteration is reached (A choice)

        Therefore if choice is chosen:
        1 -> Break base on desred threshold
        2 -> Break base on desired iteration
    */
    double step_size = 0.001;
    double x1 = 4.0, x2 = 0.5;
    double minimum = 0.0;
    double min_range = -4.5, max_range = 4.5;
    double threshold = 0.00001;
    int dimension = 2;
    int h_dimension = dimension * dimension;
    int iteration = 0;
    int max_iteration = 10000;
    int choice = 1;

    /* Create a file */
    FILE *fptr;
    fptr = fopen("export_data_from_c/result.txt", "w");
    if(fptr == NULL) {
        printf("An error has occured\n"); 
        exit(1);
    }

    /* Create arrays dynamically for f(x), gradient & hessian*/
    double *x_array, *gradient_array, *hessian_array;
    x_array = (double *)malloc(dimension * sizeof(double));
    gradient_array = (double *)malloc(dimension * sizeof(double));
    hessian_array = (double *)malloc(h_dimension * sizeof(double));

    /* Create array to store previous iteration of x_array */
    double *prev_x_array;
    prev_x_array = (double *)malloc(dimension * sizeof(double));

    /* Create array to find interval between each iteration of x_array */
    double *interval;
    interval = (double *)malloc(dimension * sizeof(double));

    /* Put in starting point into x_array */
    x_array[0] = x1;
    x_array[1] = x2;

    while(1) {
        /* Store previous information into x_array */
        memcpy(prev_x_array, x_array, (dimension * sizeof(double)));

        /* Get minimum of current iteration */
        minimum = valueandderivatives_beale2d(dimension, prev_x_array, gradient_array, hessian_array);

        /* Do the formula of plain gradient descent with step size */
        for(int index = 0; index < dimension; index++) {
            x_array[index] = prev_x_array[index] - (step_size * gradient_array[index]);
        }

        /* Print out values for current iteration*/
        fprintf(fptr, "Iteration[%10d]: x = [%f, %f] y = %f \n", iteration, x_array[0], x_array[1], minimum);

         /* Get interval for each step */
        for(int index = 0; index < dimension; index++) {
            interval[index] = fabs(x_array[index] - prev_x_array[index]);
        }

        /* Print out values of interval between x interval and x-1 interval */
        fprintf(fptr, "Interval[0]: %f Interval[1]: %f\n", interval[0], interval[0]);

        /* Break the optimzation loop if values in f(x) are found to surpass domain range */
        for(int index = 0; index < dimension; index++) {
            if((x_array[index] > max_range) || (x_array[index] < min_range)) {
                printf("Boundary has been breached\n");
                exit(1);
            }
        }

        /* Break the optimzation loop if interval has reached desired threshold */
        if(choice == 1) {
            if(interval[0] <= threshold && iteration != 0) {
                break;
            }
        }
        /* Break the optimzation loop if max iteration has been reached */
        if(choice == 2) {
            if (iteration >= max_iteration) {
                break;
            }
        }

        /* Increment iteration */
        iteration++;
    }

    /* Deallocate Memory*/
    free(x_array);
    free(prev_x_array);
    free(gradient_array);
    free(hessian_array);
}


void gradient_descent_with_momentum() {
    /*
        Optimzation Loop is breakable only:
        1. Boundary field of domain range has been reached / breached (Not a choice)
        2. Desired Threshold for intervals between iteration is satisfied (A choice)
        3. Desired Iteration is reached (A choice)

        Therefore if choice is chosen:
        1 -> Break base on desred threshold
        2 -> Break base on desired iteration
    */
    double step_size = 0.001;
    double momentum = 0.3;
    double x1 = 4.0, x2 = 0.5;
    double minimum = 0.0;
    double min_range = -4.5, max_range = 4.5;
    double threshold = 0.00001;
    int dimension = 2;
    int h_dimension = dimension * dimension;
    int iteration = 0;
    int max_iteration = 10000;
    int choice = 1;

    /* Create a file */
    FILE *fptr;
    fptr = fopen("export_data_from_c/result.txt", "w");
    if(fptr == NULL) {
        printf("An error has occured\n"); 
        exit(1);
    }

    /* Create arrays dynamically for f(x), m_array, gradient & hessian */
    double *x_array, *m_array, *gradient_array, *hessian_array;
    x_array = (double *)malloc(dimension * sizeof(double));
    m_array = (double *)malloc(dimension * sizeof(double));
    gradient_array = (double *)malloc(dimension * sizeof(double));
    hessian_array = (double *)malloc(h_dimension * sizeof(double));

    /* Create array to store previous iteration of x_array & m_array */
    double *prev_x_array, *prev_m_array;
    prev_x_array = (double *)malloc(dimension * sizeof(double));
    prev_m_array = (double *)malloc(dimension * sizeof(double));

    /* Create array to find interval between each iteration of x_array */
    double *interval;
    interval = (double *)malloc(dimension * sizeof(double));

    /* Put in starting point into x_array */
    x_array[0] = x1;
    x_array[1] = x2;

    /* Fill up m_array with 0s initially */
    for(int index = 0; index < dimension; index++) {
        m_array[index] = 0;
    }

    while(1) {
        /* Store previous information into x_array & m_array */
        memcpy(prev_x_array, x_array, (dimension * sizeof(double)));
        memcpy(prev_m_array, m_array, (dimension * sizeof(double)));

        /* Get minimum of current iteration */
        minimum = valueandderivatives_beale2d(dimension, prev_x_array, gradient_array, hessian_array);

        /* Do the formula of plain gradient descent with step size and momentum */
        for(int index = 0; index < dimension; index++) {
            m_array[index] = (momentum * prev_m_array[index]) + (step_size * gradient_array[index]);
        }
        for(int index = 0; index < dimension; index++) {
            x_array[index] = prev_x_array[index] - m_array[index];
        }

        /* Print out values for current iteration*/
        fprintf(fptr, "Iteration[%10d]: x = [%f, %f] y = %f \n", iteration, x_array[0], x_array[1], minimum);

         /* Get interval for each step */
        for(int index = 0; index < dimension; index++) {
            interval[index] = fabs(x_array[index] - prev_x_array[index]);
        }

        /* Print out values of interval between x interval and x-1 interval */
        fprintf(fptr, "Interval[0]: %f Interval[1]: %f\n", interval[0], interval[0]);

        /* Break the optimzation loop if values in f(x) are found to surpass domain range */
        for(int index = 0; index < dimension; index++) {
            if((x_array[index] > max_range) || (x_array[index] < min_range)) {
                printf("Boundary has been breached\n");
                exit(1);
            }
        }

        /* Break the optimzation loop if interval has reached desired threshold */
        if(choice == 1) {
            if(interval[0] <= threshold && iteration != 0) {
                break;
            }
        }
        /* Break the optimzation loop if max iteration has been reached */
        if(choice == 2) {
            if (iteration >= max_iteration) {
                break;
            }
        }

        /* Increment iteration */
        iteration++;
    }

    /* Deallocate Memory*/
    free(x_array);
    free(prev_x_array);
    free(gradient_array);
    free(hessian_array);
}


double valueonly_beale2d(int dim, double *x) {
  if (dim !=2) {
    printf("dim is not 2, but %d\n",dim);
    exit(2);
  }

  double p1,p2,p3;

  p1= 1.5 - x[0] +x[0]*x[1];
  p2= 2.25 - x[0] +x[0]*x[1]*x[1];
  p3= 2.625 - x[0] +x[0]*x[1]*x[1]*x[1];

  double ret = p1*p1 + p2*p2 + p3*p3;

  return ret;
}


double valueandderivatives_beale2d(int dim, double *x , double* grad, double *hessian_vecshaped) {
  if (dim !=2) {
    printf("dim is not 2, but %d\n",dim);
    exit(2);
  }

  if (grad == NULL) {
    printf("valueandderivatives_beale2d: grad == NULL\n");
    exit(10);
  }
  if (hessian_vecshaped == NULL) {
    printf("valueandderivatives_beale2d: hessian_vecshaped == NULL\n");
    exit(11);
  }

  double ret  = valueonly_beale2d( dim, x);

  double p1,p2,p3;


  //gradient
  p1= 1.5 - x[0] +x[0]*x[1];
  p2= 2.25 - x[0] +x[0]*x[1]*x[1];
  p3= 2.625 - x[0] +x[0]*x[1]*x[1]*x[1];

  grad[0] = 2*p1*(-1+x[1]) + 2*p2*(-1+x[1]*x[1])  + 2*p3*(-1+x[1]*x[1]*x[1]);
  grad[1] = 2*p1*x[0] +  2*p2*2*x[0]*x[1] + 2*p3*3*x[0]*x[1]*x[1];

  //Hessian
  double q1,q2,q3;
  q1 = -1+x[1];
  q2 = -1+x[1]*x[1];
  q3 = -1+x[1]*x[1] *x[1];

  hessian_vecshaped[0+2*0] = 2*q1*q1 + 2*q2*q2 + 2*q3*q3;
  hessian_vecshaped[1+2*1] = 2*x[0]*x[0]
                          + 8*x[0]*x[0]*x[1]*x[1] + 2*p2*2*x[0]
                          + 18*x[0]*x[0]*x[1]*x[1]*x[1]*x[1] + 2*p3*6*x[0]*x[1];

  hessian_vecshaped[1+2*0] = 2*x[0]*q1 +2*p1 + 4*x[0]*x[1]*q2 + 2*p2*2*x[1]
                          + 6*x[0]*x[1]*x[1]*q3 + 2*p3*3*x[1]*x[1];
  hessian_vecshaped[0+2*1] = hessian_vecshaped[1+2*0];
  return ret;

}
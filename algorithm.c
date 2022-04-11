#include <math.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>


void menu();
void option_1();
void option_2();
void option_3();
void gradient_descent_with_k(double* k, double *X1, double *X2, double *min, double *max, double *in_threshold, int *dim, int *max_itr, int* choice);
void gradient_descent_with_momentum(double* k, double* m, double *X1, double *X2, double *min, double *max, double *in_threshold, int *dim, int *max_itr, int* in_choice);
void newtons_algorithm();
double valueonly_beale2d(int dim, double* x);
double valueandderivatives_beale2d(int dim, double* x, double* grad, double* hessian_vecshaped);
gsl_matrix *invert_a_matrix(gsl_matrix *matrix, int size);
void get_range(double *min_input, double *max_input);
void get_step_size(double *step_size_input);
void get_momentum(double *momentum_input);
void get_e_stabilizer(double *e_input);
void get_dimension(int *dim_input);
void get_threshold(double *threshold_input);
void get_seed_value(int *seed_value_input);
void get_max_iteration(int *itr_input);
void get_choice_of_breakpoint(int *choice_input);


int main() {
    menu();
    return 0;
}


void menu() {
    int loop_choice = 1;
    printf("Welcome to ICT1002 Project\n");
    printf("Which algorithm do you want to run?\n");

    char x, choice;
    while (loop_choice == 1) {
        printf("1. Plain Gradient with Step Size\n");
        printf("2. Gradian Descent with a momentum term\n");
        printf("3. Newtons Algorithm\n");
        printf("Choice [1-3]: ");
        x = fgetc(stdin);
        choice = x;
        /* Discard rest of input line. */
        while(x != '\n' && x != EOF) {
            x = fgetc(stdin);
        }
        if(isalnum(choice)) {
            if(isalpha(choice)) {
                printf("It is an alphabet\n");
                printf("Please Try Again\n\n\n");
            }
            else {
                if(choice == '1') {
                    printf("Implementing Plain Gradient with Step Size\n");
                    option_1();
                    loop_choice = 0;
                }
                else if (choice == '2') {
                    printf("Implementing Gradian Descent with a momentum term\n");
                    option_2();
                    loop_choice = 0;
                }
                else if (choice == '3') {
                    printf("Implementing Newtons Algorithm\n");
                    option_3();
                    loop_choice = 0;
                }
                else{
                    printf("Incorrect Choice\n");
                    printf("Please Try Again\n\n\n");
                }
            }
        }
        else {
            printf("It is not an alphanumerical\n");
            printf("Please Try Again\n\n\n");
        }
    }
}


void option_1() {
    /* Get domain range desired */
    double max_range, min_range;
    get_range(&min_range, &max_range);
    
    /* Get Dimension */
    int dimension;
    get_dimension(&dimension);

    /* Get step_size */
    double k;
    get_step_size(&k);

    /* Get Threshold */
    double threshold;
    get_threshold(&threshold);

    /* Get Seed Value */
    int seed_value;
    get_seed_value(&seed_value);

    printf("######Plain Gradient with Step Size Algorithm######\n\n\n");

    /* Set seed generator */
    srand(seed_value);

    /* Add random starting number */
    double random_num = 0.0;
    double x1, x2;
        /* Set random number for x1*/
    random_num = ((double) rand() / RAND_MAX) * (max_range - min_range) + min_range;
    x1 = random_num * 1.0;
        /* Set random number for x2*/
    random_num = ((double) rand() / RAND_MAX) * (max_range - min_range) + min_range;
    x2 = random_num * 1.0;
    printf("x1: %f x2: %f\n", x1, x2);

    /* Set max iteration */
    int max_iteration;
    get_max_iteration(&max_iteration);

    /* Get choice of break-point*/
    int choice;
    get_choice_of_breakpoint(&choice);

    /* Execute Algorithm*/
    gradient_descent_with_k(&k, &x1, &x2, &min_range, &max_range, &threshold, &dimension, &max_iteration, &choice);
}


void option_2() {
    /* Get domain range desired */
    double max_range, min_range;
    get_range(&min_range, &max_range);
    
    /* Get Dimension */
    int dimension;
    get_dimension(&dimension);

    /* Get step_size */
    double k;
    get_step_size(&k);

    /* Get Momentum */
    double m;
    get_momentum(&m);

    /* Get Threshold */
    double threshold;
    get_threshold(&threshold);

    /* Get Seed Value */
    int seed_value;
    get_seed_value(&seed_value);

    printf("######Gradient with Momentum Algorithm######\n\n\n");

    /* Set seed generator */
    srand(seed_value);

    /* Add random starting number */
    double random_num = 0.0;
    double x1, x2;
        /* Set random number for x1*/
    random_num = ((double) rand() / RAND_MAX) * (max_range - min_range) + min_range;
    x1 = random_num * 1.0;
        /* Set random number for x2*/
    random_num = ((double) rand() / RAND_MAX) * (max_range - min_range) + min_range;
    x2 = random_num * 1.0;
    printf("x1: %f x2: %f\n", x1, x2);

    /* Set max iteration */
    int max_iteration;
    get_max_iteration(&max_iteration);

    /* Get choice of break-point*/
    int choice;
    get_choice_of_breakpoint(&choice);

    /* Execute Algorithm*/
    gradient_descent_with_momentum(&k, &m, &x1, &x2, &min_range, &max_range, &threshold, &dimension, &max_iteration, &choice);
}


void option_3() {
    /* Get domain range desired */
    double max_range, min_range;
    get_range(&min_range, &max_range);
    
    /* Get Dimension */
    int dimension;
    get_dimension(&dimension);

    /* Get step_size */
    double e;
    get_e_stabilizer(&e);

    /* Get Threshold */
    double threshold;
    get_threshold(&threshold);

    /* Get Seed Value */
    int seed_value;
    get_seed_value(&seed_value);

    printf("######Newton Algorithm######\n\n\n");

    /* Set seed generator */
    srand(seed_value);

    /* Add random starting number */
    double random_num = 0.0;
    double x1, x2;
        /* Set random number for x1*/
    random_num = ((double) rand() / RAND_MAX) * (max_range - min_range) + min_range;
    x1 = random_num * 1.0;
        /* Set random number for x2*/
    random_num = ((double) rand() / RAND_MAX) * (max_range - min_range) + min_range;
    x2 = random_num * 1.0;
    printf("x1: %f x2: %f\n", x1, x2);

    /* Set max iteration */
    int max_iteration;
    get_max_iteration(&max_iteration);

    /* Get choice of break-point*/
    int choice;
    get_choice_of_breakpoint(&choice);

    /* Execute Algorithm*/
    newtons_algorithm(&e, &x1, &x2, &min_range, &max_range, &threshold, &dimension, &max_iteration, &choice);
}


void gradient_descent_with_k(double* k, double *X1, double *X2, double *min, double *max, double *in_threshold, int *dim, int *max_itr, int* in_choice) {
    /*
        Optimzation Loop is breakable only:
        1. Boundary field of domain range has been reached / breached (Not a choice)
        2. Desired Threshold for intervals between iteration is satisfied (A choice)
        3. Desired Iteration is reached (A choice)

        Therefore if choice is chosen:
        1 -> Break base on desred threshold
        2 -> Break base on desired iteration

        step_size = 0.001;
        x1 = 4.0, x2 = 0.5;
        min_range = -4.5, max_range = 4.5;
        threshold = 0.00001;
        dimension = 2;       
    */
    double step_size = *k;
    double x1 = *X1, x2 = *X2;
    double minimum = 0.0;
    double min_range = *min, max_range = *max;
    double threshold = *in_threshold;
    int dimension = *dim;
    int h_dimension = dimension * dimension;
    int iteration = 0;
    int max_iteration = *max_itr;
    int choice = *in_choice;

    /* Create a file */
    FILE* fptr;
    fptr = fopen("export_data_from_c/result.txt", "w");
    if (fptr == NULL) {
        printf("An error has occured\n");
        exit(1);
    }

    /* Create arrays dynamically for f(x), gradient & hessian*/
    double* x_array, * gradient_array, * hessian_array;
    x_array = (double*)malloc(dimension * sizeof(double));
    gradient_array = (double*)malloc(dimension * sizeof(double));
    hessian_array = (double*)malloc(h_dimension * sizeof(double));

    /* Create array to store previous iteration of x_array */
    double* prev_x_array;
    prev_x_array = (double*)malloc(dimension * sizeof(double));

    /* Create array to find interval between each iteration of x_array */
    double* interval;
    interval = (double*)malloc(dimension * sizeof(double));

    /* Put in starting point into x_array */
    x_array[0] = x1;
    x_array[1] = x2;

    while (1) {
        /* Store previous information into x_array */
        memcpy(prev_x_array, x_array, (dimension * sizeof(double)));

        /* Get minimum of current iteration */
        minimum = valueandderivatives_beale2d(dimension, prev_x_array, gradient_array, hessian_array);

        /* Do the formula of plain gradient descent with step size */
        for (int index = 0; index < dimension; index++) {
            x_array[index] = prev_x_array[index] - (step_size * gradient_array[index]);
        }

        /* Print out values for current iteration*/
        fprintf(fptr, "Iteration[%d]: x = [%f, %f] y = %f\n", iteration, x_array[0], x_array[1], minimum);

        /* Get interval for each step */
        for (int index = 0; index < dimension; index++) {
            interval[index] = fabs(x_array[index] - prev_x_array[index]);
        }

        /* Break the optimzation loop if values in f(x) are found to surpass domain range */
        for (int index = 0; index < dimension; index++) {
            if ((x_array[index] > max_range) || (x_array[index] < min_range)) {
                printf("Boundary has been breached\n");
                exit(1);
            }
        }

        /* Break the optimzation loop if interval has reached desired threshold */
        if (choice == 1) {
            if (interval[0] <= threshold && iteration != 0) {
                break;
            }
        }
        /* Break the optimzation loop if max iteration has been reached */
        if (choice == 2) {
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


void gradient_descent_with_momentum(double* k, double* m, double *X1, double *X2, double *min, double *max, double *in_threshold, int *dim, int *max_itr, int* in_choice) {
    /*
        Optimzation Loop is breakable only:
        1. Boundary field of domain range has been reached / breached (Not a choice)
        2. Desired Threshold for intervals between iteration is satisfied (A choice)
        3. Desired Iteration is reached (A choice)

        Therefore if choice is chosen:
        1 -> Break base on desred threshold
        2 -> Break base on desired iteration
    */
    double step_size = *k;
    double momentum = *m;
    double x1 = *X1, x2 = *X2;
    double minimum = 0.0;
    double min_range =*min, max_range = *max;
    double threshold = *in_threshold;
    int dimension = *dim;
    int h_dimension = dimension * dimension;
    int iteration = 0;
    int max_iteration = *max_itr;
    int choice = *in_choice;

    /* Create a file */
    FILE* fptr;
    fptr = fopen("export_data_from_c/result2.txt", "w");
    if (fptr == NULL) {
        printf("An error has occured\n");
        exit(1);
    }

    /* Create arrays dynamically for f(x), m_array, gradient & hessian */
    double* x_array, * m_array, * gradient_array, * hessian_array;
    x_array = (double*)malloc(dimension * sizeof(double));
    m_array = (double*)malloc(dimension * sizeof(double));
    gradient_array = (double*)malloc(dimension * sizeof(double));
    hessian_array = (double*)malloc(h_dimension * sizeof(double));

    /* Create array to store previous iteration of x_array & m_array */
    double* prev_x_array, * prev_m_array;
    prev_x_array = (double*)malloc(dimension * sizeof(double));
    prev_m_array = (double*)malloc(dimension * sizeof(double));

    /* Create array to find interval between each iteration of x_array */
    double* interval;
    interval = (double*)malloc(dimension * sizeof(double));

    /* Put in starting point into x_array */
    x_array[0] = x1;
    x_array[1] = x2;

    /* Fill up m_array with 0s initially */
    for (int index = 0; index < dimension; index++) {
        m_array[index] = 0;
    }

    while (1) {
        /* Store previous information into x_array & m_array */
        memcpy(prev_x_array, x_array, (dimension * sizeof(double)));
        memcpy(prev_m_array, m_array, (dimension * sizeof(double)));

        /* Get minimum of current iteration */
        minimum = valueandderivatives_beale2d(dimension, prev_x_array, gradient_array, hessian_array);

        /* Do the formula of plain gradient descent with step size and momentum */
        for (int index = 0; index < dimension; index++) {
            m_array[index] = (momentum * prev_m_array[index]) + (step_size * gradient_array[index]);
        }
        for (int index = 0; index < dimension; index++) {
            x_array[index] = prev_x_array[index] - m_array[index];
        }

        /* Print out values for current iteration*/
        fprintf(fptr, "Iteration[%d]: x = [%f, %f] y = %f\n", iteration, x_array[0], x_array[1], minimum);

        /* Get interval for each step */
        for (int index = 0; index < dimension; index++) {
            interval[index] = fabs(x_array[index] - prev_x_array[index]);
        }

        /* Break the optimzation loop if values in f(x) are found to surpass domain range */
        for (int index = 0; index < dimension; index++) {
            if ((x_array[index] > max_range) || (x_array[index] < min_range)) {
                printf("Boundary has been breached\n");
                exit(1);
            }
        }

        /* Break the optimzation loop if interval has reached desired threshold */
        if (choice == 1) {
            if (interval[0] <= threshold && iteration != 0) {
                break;
            }
        }
        /* Break the optimzation loop if max iteration has been reached */
        if (choice == 2) {
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


void newtons_algorithm(double* in_e, double *X1, double *X2, double *min, double *max, double *in_threshold, int *dim, int *max_itr, int* in_choice) {
    /*
        Optimzation Loop is breakable only:
        1. Boundary field of domain range has been reached / breached (Not a choice)
        2. Desired Threshold for intervals between iteration is satisfied (A choice)
        3. Desired Iteration is reached (A choice)

        Therefore if choice is chosen:
        1 -> Break base on desred threshold
        2 -> Break base on desired iteration
    */
    double x1 = *X1, x2 = *X2;
    double e = *in_e;
    double minimum = 0.0;
    double min_range = *min, max_range = *max;
    double threshold = *in_threshold;
    int dimension = *dim;
    int h_dimension = dimension * dimension;
    int iteration = 0;
    int max_iteration = *max_itr;
    int choice = *in_choice;

    /* Create a file */
    FILE* fptr;
    fptr = fopen("export_data_from_c/result3.txt", "w");
    if (fptr == NULL) {
        printf("An error has occured\n");
        exit(1);
    }

    /* Create arrays dynamically for f(x), gradient & hessian */
    double* x_array, * gradient_array, * hessian_array;
    x_array = (double*)malloc(dimension * sizeof(double));
    gradient_array = (double*)malloc(dimension * sizeof(double));
    hessian_array = (double*)malloc(h_dimension * sizeof(double));

    /* Create array to store previous iteration of x_array */
    double* prev_x_array;
    prev_x_array = (double*)malloc(dimension * sizeof(double));

    /* Create array to find interval between each iteration of x_array */
    double* interval;
    interval = (double*)malloc(dimension * sizeof(double));

    /* Put in starting point into x_array */
    x_array[0] = x1;
    x_array[1] = x2;

    int row, col;
    /* Create constant matrix of [Identity Matrix multiply by e-stabilizer] */
    gsl_matrix* identity_matrix = gsl_matrix_alloc(dimension, dimension);
    gsl_matrix_set_identity(identity_matrix);
    for(row = 0; row < dimension; row++) {
        for(col = 0; col < dimension; col++) {
            gsl_matrix_set(identity_matrix, row, col, e * gsl_matrix_get(identity_matrix, row, col));
        }
    }

    /* Create gradient matrix [gradient_matrix] */
    gsl_matrix* gradient_matrix = gsl_matrix_alloc(1, dimension);
    for(col = 0; col < dimension; col++) {
        gsl_matrix_set(gradient_matrix, 0, col, 0);
    }

    /* Create Hessian matrix [hessian_matrix] */
    gsl_matrix* hessian_matrix = gsl_matrix_alloc(dimension, dimension);
    for(row = 0; row < dimension; row++) {
        for(col = 0; col < dimension; col++) {
            gsl_matrix_set(hessian_matrix, row, col, 0);
        }
    }

    /* Create A_matrix and inverted_A_matrix */
    gsl_matrix* A_matrix = gsl_matrix_alloc(dimension, dimension);
    gsl_matrix* inverted_A_matrix = gsl_matrix_alloc(dimension, dimension);

    /* Create B_matrix that stores result from multiplication of inverted_A_matrix and gradient_array */
    gsl_matrix* B_matrix = gsl_matrix_alloc(dimension, 1);

    while(1) {
        /* Store previous information into prev_x_array */
        memcpy(prev_x_array, x_array, (dimension * sizeof(double)));

        /* Get minimum of current iteration */
        minimum = valueandderivatives_beale2d(dimension, prev_x_array, gradient_array, hessian_array);

        /* Store information from gradient_array to gradient_matrix */
        for(col = 0; col < dimension; col++) {
            gsl_matrix_set(gradient_matrix, 0, col, gradient_array[col]);
        }

        /* Store information from hessian_array to hessian_matrix */
        int count = 0;
        for(row = 0; row < dimension; row++) {
            for(col = 0; col < dimension; col++) {
                gsl_matrix_set(hessian_matrix, row, col, hessian_array[count]);
                count++;
            }
        }

        /* Do the formula of newtons algorithm */
        /* Add Hessian Matrix and Identitfy Matix w e-stabilizer to A Matrix */
        gsl_matrix_add(hessian_matrix, identity_matrix);
        gsl_matrix_memcpy(A_matrix, hessian_matrix);
        /* Invert Matrix */
        inverted_A_matrix = invert_a_matrix(A_matrix, dimension);
        /* Muliply inverted_A_matrix and grad_array into B_matrix */
        for(row = 0; row < dimension; row++) {
            double total = 0;
            for(col = 0; col < dimension; col++) {
                total = (gsl_matrix_get(inverted_A_matrix, row, col) * gradient_array[col]) + total;
            }
            gsl_matrix_set(B_matrix, row, 0, total);
        }
        /* Use prev_x_array to minus of B_matrix into x_array */
        for(int index = 0; index < dimension; index++) {
            row = index;
            x_array[index] = prev_x_array[index] - gsl_matrix_get(B_matrix, row, 0);
        }
        /* Print out values for current iteration*/
        fprintf(fptr, "Iteration[%d]: x = [%f, %f] y = %f\n", iteration, x_array[0], x_array[1], minimum);

        /* Get interval for each step */
        for (int index = 0; index < dimension; index++) {
            interval[index] = fabs(x_array[index] - prev_x_array[index]);
        }

        /* Break the optimzation loop if values in f(x) are found to surpass domain range */
        for (int index = 0; index < dimension; index++) {
            if ((x_array[index] > max_range) || (x_array[index] < min_range)) {
                printf("Boundary has been breached\n");
                exit(1);
            }
        }

        /* Break the optimzation loop if interval has reached desired threshold */
        if (choice == 1) {
            if (interval[0] <= threshold && iteration != 0) {
                break;
            }
        }
        /* Break the optimzation loop if max iteration has been reached */
        if (choice == 2) {
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
    gsl_matrix_free(identity_matrix);
    gsl_matrix_free(gradient_matrix);
    gsl_matrix_free(hessian_matrix);
    gsl_matrix_free(A_matrix);
    gsl_matrix_free(inverted_A_matrix);
    gsl_matrix_free(B_matrix);
}


double valueonly_beale2d(int dim, double* x) {
    if (dim != 2) {
        printf("dim is not 2, but %d\n", dim);
        exit(2);
    }

    double p1, p2, p3;

    p1 = 1.5 - x[0] + x[0] * x[1];
    p2 = 2.25 - x[0] + x[0] * x[1] * x[1];
    p3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];

    double ret = p1 * p1 + p2 * p2 + p3 * p3;

    return ret;
}


double valueandderivatives_beale2d(int dim, double* x, double* grad, double* hessian_vecshaped) {
    if (dim != 2) {
        printf("dim is not 2, but %d\n", dim);
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

    double ret = valueonly_beale2d(dim, x);

    double p1, p2, p3;


    //gradient
    p1 = 1.5 - x[0] + x[0] * x[1];
    p2 = 2.25 - x[0] + x[0] * x[1] * x[1];
    p3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];

    grad[0] = 2 * p1 * (-1 + x[1]) + 2 * p2 * (-1 + x[1] * x[1]) + 2 * p3 * (-1 + x[1] * x[1] * x[1]);
    grad[1] = 2 * p1 * x[0] + 2 * p2 * 2 * x[0] * x[1] + 2 * p3 * 3 * x[0] * x[1] * x[1];

    //Hessian
    double q1, q2, q3;
    q1 = -1 + x[1];
    q2 = -1 + x[1] * x[1];
    q3 = -1 + x[1] * x[1] * x[1];

    hessian_vecshaped[0 + 2 * 0] = 2 * q1 * q1 + 2 * q2 * q2 + 2 * q3 * q3;
    hessian_vecshaped[1 + 2 * 1] = 2 * x[0] * x[0]
        + 8 * x[0] * x[0] * x[1] * x[1] + 2 * p2 * 2 * x[0]
        + 18 * x[0] * x[0] * x[1] * x[1] * x[1] * x[1] + 2 * p3 * 6 * x[0] * x[1];

    hessian_vecshaped[1 + 2 * 0] = 2 * x[0] * q1 + 2 * p1 + 4 * x[0] * x[1] * q2 + 2 * p2 * 2 * x[1]
        + 6 * x[0] * x[1] * x[1] * q3 + 2 * p3 * 3 * x[1] * x[1];
    hessian_vecshaped[0 + 2 * 1] = hessian_vecshaped[1 + 2 * 0];
    return ret;

}


gsl_matrix *invert_a_matrix(gsl_matrix *matrix, int size) {
    gsl_permutation *p = gsl_permutation_alloc(size);
    int s;

    // Compute the LU decomposition of this matrix
    gsl_linalg_LU_decomp(matrix, p, &s);

    // Compute the  inverse of the LU decomposition
    gsl_matrix *inv = gsl_matrix_alloc(size, size);
    gsl_linalg_LU_invert(matrix, p, inv);

    gsl_permutation_free(p);

    return inv;
}


void get_range(double *min_input, double *max_input) {
    int BUFFERSIZE = 100;
    int is_it_a_num = 1;
    int loop_choice;
    double  min, max;
    char choice[BUFFERSIZE];

    printf("Let's get your Domain Range [min, max]\n");

    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt [min]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' or '.' */
            if((choice[x] != '-') && (choice[x] != '.')) {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])){
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0){
                        is_it_a_num = 1;
                    }
                    else {
                        printf("min contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("min contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            char *ptr;
            min = strtod(choice, &ptr);
            loop_choice = 0;
        }
    }

    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt [max]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' or '.' */
            if((choice[x] != '-') && (choice[x] != '.')) {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])) {
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0) {
                        is_it_a_num = 1;
                    }
                    else {
                        printf("max contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("max contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            char *ptr;
            max = strtod(choice, &ptr);
            loop_choice = 0;
        }
    }

    *min_input = min;
    *max_input = max;
}


void get_step_size(double *step_size_input) {
    int BUFFERSIZE = 100, is_it_a_num = 1, loop_choice;
    char choice[BUFFERSIZE];
    double step_size;

    printf("What is the step size you require?\n");

    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt[Step Size]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' */
            if((choice[x] != '-') && (choice[x] != '.')) {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])){
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0){
                        is_it_a_num = 1;
                    }
                    else {
                        printf("Step Size contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("Step Size contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            char *ptr;
            step_size = strtod(choice, &ptr);
            loop_choice = 0;
        }
    }

    *step_size_input = step_size;
}


void get_momentum(double *momentum_input) {
    int BUFFERSIZE = 100, is_it_a_num = 1, loop_choice;
    char choice[BUFFERSIZE];
    double momentum;

    printf("What is the momentum you require? [Between 1 to 0]\n");

    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt[Momentum]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' */
            if((choice[x] != '-') && (choice[x] != '.')) {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])){
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0){
                        is_it_a_num = 1;
                    }
                    else {
                        printf("Momentum contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("Momentum contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            char *ptr;
            momentum = strtod(choice, &ptr);
            if((momentum >= 0) && (momentum <= 1)) {
                loop_choice = 0;
            }
            else {
                printf("Choice is not allowed\n");
            }
        }
    }

    *momentum_input = momentum;
}


void get_e_stabilizer(double *e_input) {
    int BUFFERSIZE = 100, is_it_a_num = 1, loop_choice;
    char choice[BUFFERSIZE];
    double e;

    printf("What is the e-stabilizer you require? [Between 0.001 to 0.000001]\n");

    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt[e-stabilizer]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' */
            if((choice[x] != '-') && (choice[x] != '.')) {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])){
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0){
                        is_it_a_num = 1;
                    }
                    else {
                        printf("e-stabilizer contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("e-stabilizer contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            char *ptr;
            e = strtod(choice, &ptr);
            if((e >= 0.000001) && (e <= 0.001)) {
                loop_choice = 0;
            }
            else {
                printf("Choice is not allowed\n");
            }
        }
    }

    *e_input = e;
}


void get_dimension(int *dim_input) {
    int BUFFERSIZE = 100, is_it_a_num = 1, dimension, loop_choice;
    char choice[BUFFERSIZE];
    printf("What is the dimension you require?\n");
    loop_choice = 1;
    while (loop_choice == 1) {
        fputs("Prompt [Dimension]: ", stdout);
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' */
            if(choice[x] != '-') {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])){
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0){
                        is_it_a_num = 1;
                    }
                    else {
                        printf("Dimension contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("Dimension contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            dimension = atoi(choice);
            loop_choice = 0;
        }
    }
    *dim_input = dimension;
}


void get_threshold(double *threshold_input) {
    int BUFFERSIZE = 100, is_it_a_num = 1, loop_choice;
    char choice[BUFFERSIZE];
    double threshold;

    printf("What is the threshold you require? [Example: 0.001]\n");

    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt[Threshold]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' */
            if((choice[x] != '-') && (choice[x] != '.')) {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])){
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0){
                        is_it_a_num = 1;
                    }
                    else {
                        printf("Threshold contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("Threshold contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            char *ptr;
            threshold = strtod(choice, &ptr);
            loop_choice = 0;
        }
    }

    *threshold_input = threshold;
}


void get_seed_value(int *seed_value_input) {
    int BUFFERSIZE = 100, is_it_a_num = 1, seed_value, loop_choice;
    char choice[BUFFERSIZE];

    printf("What is the seed value you require?\n");

    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt [Seed Value]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a alphanumerical */
            if(isalnum(choice[x])){
                /* Check if character is an alphabet */
                if(isalpha(choice[x]) == 0){
                    is_it_a_num = 1;
                }
                else {
                    printf("Seed Value contains an alphabet\n");
                    is_it_a_num = 0;
                    break;
                }
            }
            else {
                printf("Seed Value contains a non-alphanumerical\n");
                is_it_a_num = 0;
                break;
            }
        }
        if(is_it_a_num) {
            seed_value = atoi(choice);
            loop_choice = 0;
        }
    }
    *seed_value_input = seed_value;
}


void get_max_iteration(int *itr_input) {
    int BUFFERSIZE = 100, is_it_a_num = 1, itr, loop_choice;
    char choice[BUFFERSIZE];

    printf("What is the max iteration you require?\n");

    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt [Iteration 1 - 100000]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' */
            if(choice[x] != '-') {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])){
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0){
                        is_it_a_num = 1;
                    }
                    else {
                        printf("Iteration contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("Iteration contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            itr = atoi(choice);
            loop_choice = 0;
        }
    }
    *itr_input = itr;
}


void get_choice_of_breakpoint(int *choice_input) {
    int BUFFERSIZE = 100, is_it_a_num = 1, input, loop_choice;
    char choice[BUFFERSIZE];

    printf("What is the choice you require?\n");
    printf("1 -> Break base on desred threshold\n");
    printf("2 -> Break base on desired iteration\n");
    loop_choice = 1;
    while (loop_choice == 1) {
        printf("Prompt [1 or 2]: ");
        fgets(choice, BUFFERSIZE, stdin);
        int iteration = strlen(choice) - 1;
        for(int x = 0; x < iteration; x++) {
            /* Check if character is a '-' */
            if(choice[x] != '-') {
                /* Check if character is a alphanumerical */
                if(isalnum(choice[x])){
                    /* Check if character is an alphabet */
                    if(isalpha(choice[x]) == 0){
                        is_it_a_num = 1;
                    }
                    else {
                        printf("Choice contains an alphabet\n");
                        is_it_a_num = 0;
                        break;
                    }
                }
                else {
                    printf("Choice contains a non-alphanumerical\n");
                    is_it_a_num = 0;
                    break;
                }
            }
        }
        if(is_it_a_num) {
            input = atoi(choice);
            if((input > 0) && (input < 3)) {
                loop_choice = 0;
            }
            else {
                printf("Choice is not allowed\n");
            }
        }
    }
    *choice_input = input;
}



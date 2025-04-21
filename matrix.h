#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
struct matrix {
    double *mat;
    int r, c;
};

struct vector {
    double *vec;
    int n;
};

constexpr double sigmoid(double x) {
    return 1/(1 + exp(-x));
}
constexpr double sigmoid_derivative(double x) {
    const double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}
constexpr void sigmoid(matrix x) {
    for (int i = 0; i < x.r * x.c; ++i) {
        x.mat[i] = sigmoid(x.mat[i]);
    }
}
constexpr void sigmoid_derivative(matrix x) {
    for (int i = 0; i < x.r * x.c; ++i) {
        x.mat[i] = sigmoid_derivative(x.mat[i]);
    }
}

double at(matrix m, int r, int c);
double at_t(matrix m, int r, int c);
double at(vector v, int i);

matrix matmul(matrix a, matrix b);
matrix matmul_tl(matrix a, matrix b);
matrix matmul_tr(matrix a, matrix b);
matrix compute_error(matrix target, matrix current);
matrix update_weights(matrix errors, matrix ok, matrix oj);

matrix zero_matrix(int r, int c);
matrix rand_matrix(int r, int c);
void print_matrix(matrix m);

#endif /* ifndef MATRIX_H */

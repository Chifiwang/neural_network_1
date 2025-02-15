#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include <cmath>

#define INPUT_SIZE 6
#define HIDDEN_SIZE 5
#define OUTPUT_SIZE 10

void propagate_forwards(matrix weights, vector in, gradient_data out, int in_len, int out_len);
void propagate_backwards(matrix weights, vector error, gradient_data dat, int a_r, int a_c,  int error_len, int grad_len);
constexpr double sigmoid(double x) {
    return 1/(1 + exp(-x));
}
constexpr double sigmoid_derivative(double x) {
    const double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}
constexpr void sigmoid(vector x, int len) {
    for (int i = 0; i < len; ++i) {
        x[i] = sigmoid(x[i]);
    }
}
constexpr void sigmoid_derivative(vector x, int len) {
    for (int i = 0; i < len; ++i) {
        x[i] = sigmoid_derivative(x[i]);
    }
}


#endif /* ifndef LAYER_H */

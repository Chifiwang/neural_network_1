#include "layer.h"
#include "matrix.h"
#include "neural_network.h"
#include <cstring>

void propagate_forwards(matrix weights, vector in, gradient_data out, int in_len, int out_len) {
    // printf("test\n");
    mat_vec_mul(weights, in, out.vals, in_len, out_len);
    // memcpy(out.deriv, out.vals, sizeof(double) * out_len);
    // for (int i = 0; i < out_len;++i) {
    //     printf("%f ", out.vals[i]);
    // } printf("\n");
    sigmoid(out.vals, out_len);
    for (int j = 0; j < out_len; ++j) {
        out.deriv[j] = out.vals[j] * (1 - out.vals[j]) * in[j]; // sigmoid derivative
    }
}

void propagate_backwards(matrix weights, vector error, gradient_data dat, int a_r, int a_c,  int error_len, int grad_len) {
    // printf("HERE\n");
    for (int i = 0; i < a_r; ++i) {
        for (int j = 0; j < a_c; ++j) {
            // printf("%f %f %f\n", weights[index(i, j, a_r)], dat.vals[j], error[j]);
            weights[index(i, j, a_r)] = weights[index(i, j, a_r)] + LEARNING_RATE * error[j] * dat.deriv[j];
        }
    }
}

#include "matrix.h"
#include <cassert>

void mat_vec_mul(matrix a, vector x, vector out, int a_r, int a_c, int x_len, int out_len) {
    assert(a_c == x_len);
    assert(a_r == out_len);

    for (int r = 0; r < a_r; ++r) {
        for (int c = 0; c < a_c; ++c) {
            out[r] += a[index(r, c, a_r)] * x[c];
        }
    }
}

void mat_vec_mul(matrix a, vector in, vector out, int in_len, int out_len) {
    mat_vec_mul(a, in, out, out_len, in_len, in_len, out_len);
}

void mat_T_vec_mul(matrix a, vector x, vector out, int a_r, int a_c, int x_len, int out_len) {
    assert(a_r == x_len);
    assert(a_c == out_len);

    for (int c = 0; c < a_c; ++c) {
        for (int r = 0; r < a_r; ++r) {
            out[c] = a[index_t_inv(r, c, a_r, a_c)];
        }
    }
}

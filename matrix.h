#ifndef MATRIX_H
#define MATRIX_H

// typedef std::vector<std::vector<double>> matrix;
typedef double *matrix;
typedef double *vector;
struct gradient_data {
    vector vals;
    vector deriv;
};

// constexpr matrix mat_mul(matrix &a, matrix &b);
void mat_vec_mul(matrix a, vector x, vector out, int a_r, int a_c, int x_len, int out_len);
void mat_vec_mul(matrix a, vector in, vector out, int in_len, int out_len);
void mat_T_vec_mul(matrix a, vector x, vector out, int a_r, int a_c, int x_len, int out_len);
void mat_T_vec_mul(matrix a, vector in, vector out, int in_len, int out_len);
constexpr int index(int r, int c, int r_max) { return r * r_max + c; }
constexpr int index_t(int r, int c, int a_r, int a_c) {
    const int idx = index(r, c, a_r);
    const int C = a_r * a_c - 1;
    if (idx == C) {
        return idx;
    }

    return (a_r * idx) % C;
}
constexpr int index_t_inv(int r, int c, int a_r, int a_c) {
    const int idx = index(c, r, a_c);
    const int C = a_r * a_c - 1;
    if (idx == C) {
        return idx;
    }

    return (a_c * idx) % C;
}


#endif /* ifndef MATRIX_H */

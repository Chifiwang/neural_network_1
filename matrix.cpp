#include "matrix.h"
#include "random.h"
#include <iostream>
#include <cassert>

double at(matrix m, int r, int c) {
    return m.mat[r * m.c + c];
}

double at_t(matrix m, int r, int c) {
    return m.mat[c * m.c + r];
}

double at(vector v, int i) {
    return v.vec[i];
}

matrix matmul(matrix a, matrix b) {
    matrix ab = matrix{new double[a.r * b.c], a.r, b.c};

    for (int i = 0; i < a.r; ++i) {
        for (int j = 0; j < b.c; ++j) {
            ab.mat[i * ab.c + j] = 0;
            for (int k = 0; k < b.r; ++k) {
                // printf("%d %d %f\n", i, k, at(a, i, k));
                ab.mat[i * ab.c + j] += at(a, i, k) * at(b, k, j);
            }
            // printf("\n%f\n", at(ab, i, j));
        }
    }

    return ab;
}

matrix matmul_tl(matrix aT, matrix b) {
    matrix aTb = matrix{new double[aT.c * b.c], aT.c, b.c};

    for (int i = 0; i < aT.c; ++i) {
        for (int j = 0; j < b.c; ++j) {
            aTb.mat[i * aTb.c + j] = 0;
            for (int k = 0; k < b.r; ++k) {
                // printf("%d %d %f\n", i, k, at_t(aT, i, k));

                aTb.mat[i * aTb.c + j] += at_t(aT, i, k) * at(b, k, j);
            }

            // printf("\n%f\n", at(aTb, i, j));
        }
    }

    return aTb;
}

matrix matmul_tr(matrix a, matrix bT) {
    matrix abT = matrix{new double[a.r * bT.r], a.r, bT.r};

    for (int i = 0; i < a.r; ++i) {
        for (int j = 0; j < bT.r; ++j) {
            abT.mat[i * abT.c + j] = 0;
            for (int k = 0; k < bT.c; ++k) {
                // printf("%d %d %f\n", i, k, at_t(aT, i, k));

                abT.mat[i * abT.c + j] += at(a, i, k) * at_t(bT, k, j);
            }

            // printf("\n%f\n", at(aTb, i, j));
        }
    }

    return abT;
}


matrix compute_error(matrix a, matrix b) {
    matrix error = matrix{new double[b.r * b.c], b.r, b.c};
    for (int i = 0; i < a.r * a.c; ++i) {
        error.mat[i] = a.mat[i] - b.mat[i];
    }

    return error;
}

matrix update_weights(matrix errors, matrix ok, matrix oj) {
    for (int i = 0; i < ok.r * ok.c; ++i) {
        ok.mat[i] = errors.mat[i] * sigmoid_derivative(ok.mat[i]);
    }

    return matmul_tl(ok, oj);
}

matrix zero_matrix(int r, int c) {
    return matrix{new double[r * c](), r, c};
}

matrix rand_matrix(int r, int c) {
    matrix mat = matrix{new double[r * c], r, c};

    double range = std::sqrt(6)/(r + c);
    for (int i = 0; i < r * c; ++i) {
        mat.mat[i] = Random::get_rand<double>(-range, range);
    }
    // NOTE: Identity matrix gen for now
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            if (i == j) {
                mat.mat[i * mat.c + j] = 1;
            } else {
                mat.mat[i * mat.c + j] = 0;
            }
        }
    }

    return mat;
}

void print_matrix(matrix m) {
    for (int i = 0; i < m.r; ++i) {
        for (int j = 0; j < m.c; ++j) {
            std::cout << at(m, i, j) << ' ';
        }
        std::cout << '\n';
    }
}

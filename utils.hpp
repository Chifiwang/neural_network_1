#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <iostream>
#include "random.hpp"

struct matrix {
    double **mat;
    int r, c;

    constexpr double*& operator[](int i) {
        return mat[i];
    }
};

struct vector {
    double *vec;
    int n;

    constexpr double& operator[](int i) {
        return vec[i];
    }
};

constexpr double sigmoid(double x) {
    double y = 1/(1 + exp(-x));
    // printf("%f %f \n", x, y);
    return y;
}
constexpr double sigmoid_derivative(double x) {
    const double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}
constexpr void sigmoid(matrix x) {
    for (int i = 0; i < x.r; ++i) {
        for (int j = 0; j < x.c; ++j) {
            x.mat[i][j] = sigmoid(x.mat[i][j]);
        }
    }
}
constexpr void sigmoid_derivative(matrix x) {
    for (int i = 0; i < x.r; ++i) {
        for (int j = 0; j < x.c; ++j) {
            x.mat[i][j] = sigmoid_derivative(x.mat[i][j]);
        }
    }
}


inline vector zero_vector(int n) {
    return vector{new double[n](), n};
}

inline matrix zero_matrix(int r, int c) {
    matrix m = matrix{new double*[r], r, c};
    for (int i = 0; i < r; ++i) {
        m[i] = new double[c]();
    }

    return m;
}

inline matrix rand_matrix(int r, int c) {
    matrix mat = matrix{new double*[r], r, c};

    double range = std::sqrt(6)/(r + c);
    for (int i = 0; i < r; ++i) {
        mat[i] = new double[c];
        for (int j = 0; j < c; ++j) {
            mat[i][j] = Random::get_rand<double>(-range, range);
        }
    }
    // NOTE: Identity matrix gen for now
    // for (int i = 0; i < r; ++i) {
    //     for (int j = 0; j < c; ++j) {
    //         if (i == j) {
    //             mat[i][j] = 1;
    //         } else {
    //             mat[i][j] = 0;
    //         }
    //     }
    // }

    return mat;
}

inline void print_matrix(matrix &m) {
    for (int i = 0; i < m.r; ++i) {
        for (int j = 0; j < m.c; ++j) {
            std::cout << m[i][j] << ' ';
        }
        std::cout << '\n';
    }
}

inline void print_vector(vector &v) {
    for (int i = 0; i < v.n; ++i) {
        std::cout << v[i] << ' ';
    }
    std::cout << '\n';
}

#endif /* ifndef UTILS_HPP */

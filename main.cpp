#include "layer.h"
#include "neural_network.h"
#include "matrix.h"
#include <iostream>

int main() {
    neural_network net(3);

    net.propagate_forwards();
    net.dbg_print();
    matrix mat = (matrix) malloc(sizeof(double) * 9);
    mat[index(0, 0, 3)] = 0.3;
    mat[index(0, 1, 3)] = 0.7;
    mat[index(0, 2, 3)] = 0.5;
    mat[index(1, 0, 3)] = 0.6;
    mat[index(1, 1, 3)] = 0.5;
    mat[index(1, 2, 3)] = 0.2;
    mat[index(2, 0, 3)] = 0.8;
    mat[index(2, 1, 3)] = 0.1;
    mat[index(2, 2, 3)] = 0.9;

    // vector vec = (vector) malloc(sizeof(double) * 3);
    // vec[0] = 0.761;
    // vec[1] = 0.603;
    // vec[2] = 0.650;
    //
    // gradient_data out;
    // out.vals = (vector) malloc(sizeof(double) * 3);
    // out.deriv = (vector) new double[3];
    // mat_vec_mul(mat, vec, out, 3, 3, 3, 3);
    // propagate_forwards(mat, vec, out, 3, 3);
    //
    // std::cout << out.vals[0] << ' ' << out.vals[1] << ' ' << out.vals[2] << '\n';
    // std::cout << out.deriv[0] << ' ' << out.deriv[1] << ' ' << out.deriv[2] << '\n';

    matrix mat2 = (matrix) new double[4];
    mat2[index(0, 0, 2)] = 2;
    mat2[index(0, 1, 2)] = 4;
    mat2[index(1, 0, 2)] = 3;
    mat2[index(1, 1, 2)] = 1;

    gradient_data out2;
    out2.vals = new double[2];
    out2.deriv = new double[2];

    vector in = new double[2];
    in[0] = 0.4;
    in[1] = 0.5;

    propagate_forwards(mat2, in, out2, 2, 2);
    std::cout << out2.vals[0] << ' ' << out2.vals[1] << '\n';
    std::cout << out2.deriv[0] << ' ' << out2.deriv[1] << '\n';

    vector err = new double[2];
    err[0] = 1.5;
    err[1] = 0.5;

    propagate_backwards(mat2, err, out2, 2, 2, 2, 2);
    std::cout << mat2[0] << ' ' << mat2[1] << '\n';
    std::cout << mat2[2] << ' ' << mat2[3] << '\n';
}

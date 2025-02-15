#include "neural_network.h"
#include "layer.h"
#include <iostream>

void dbg_print_mat(matrix x, int r, int c, int r_max) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << x[index(r, c, r_max)] << ' ';
        } std::cout << '\n';
    }
}

void neural_network::dbg_print() {
    dbg_print_mat(layers_m[0], HIDDEN_SIZE, INPUT_SIZE, INPUT_SIZE);
    std::cout << '\n';
    for (int i = 1; i < num_hidden_layers; ++i) {
        dbg_print_mat(layers_m[i], HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE);
        std::cout << '\n';
    }
    dbg_print_mat(layers_m[num_hidden_layers], OUTPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE);
}

void neural_network::propagate_forwards() {
    ::propagate_forwards(layers_m[0], input_m, buf_m[1], INPUT_SIZE, HIDDEN_SIZE);
    for (int i = 1; i < num_hidden_layers - 1; ++i) {
        ::propagate_forwards(layers_m[i], buf_m[i].vals, buf_m[i+1], HIDDEN_SIZE, HIDDEN_SIZE);
    }
    ::propagate_forwards(layers_m[num_hidden_layers], buf_m[num_hidden_layers].vals, buf_m[num_hidden_layers + 1], HIDDEN_SIZE, OUTPUT_SIZE);
}

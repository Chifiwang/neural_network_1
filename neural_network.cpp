#include "neural_network.h"
#include "layer.h"
#include <iostream>

void neural_network::dbg_print() {
    for (int i = 0; i < INPUT_SIZE; ++i) {
        std::cout << layers_m[0][i] << ' ';
    } std::cout << '\n';


    for (int j = 1; j <= num_hidden_layers; ++j) {
        for (int i = 0; i < INPUT_SIZE; ++i) {
            std::cout << layers_m[j][i] << ' ';
        } std::cout << '\n';
    }

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        std::cout << layers_m[num_hidden_layers + 1][i] << ' ';
    } std::cout << '\n';
}

constexpr double neural_network::sigmoid(double x) {
    return 1/(1 + exp(-x));
}

constexpr double neural_network::sigmoid_derivative(double x) {
    const double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "layer.h"
#include <cstddef>
#include <stdio.h>
#include <vector>

#define LEARNING_RATE 0.2

typedef std::vector<double *> network_layers;

class neural_network {
    network_layers layers_m;
    const unsigned int num_hidden_layers;

public:
    neural_network(unsigned int num_hidden_layers) : num_hidden_layers{num_hidden_layers} {
        layers_m = std::vector<double *>(num_hidden_layers + 2);

        layers_m[0] = (double *) malloc(sizeof(double *) * INPUT_SIZE);
        for (int i = 1; i < num_hidden_layers + 1; ++i) {
            layers_m[i] = (double *) malloc(sizeof(double *) * HIDDEN_SIZE);
        }
        layers_m[num_hidden_layers + 1] = (double *) malloc(sizeof(double *) * OUTPUT_SIZE);
    }

    constexpr double sigmoid(double x);
    constexpr double sigmoid_derivative(double x);

    // DEBUG
    void dbg_print();
};

#endif /* ifndef NEURAL_NETWORK_H */

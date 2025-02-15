#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "layer.h"
#include "matrix.h"
#include <cstddef>
#include <iostream>
#include <stdio.h>
#include <vector>

#define LEARNING_RATE 0.1

typedef std::vector<matrix> network_layers;
typedef std::vector<gradient_data> network_buffers;

class neural_network {
public:
    vector input_m;
    vector output_m;
    network_layers layers_m;
    network_buffers buf_m;
    const unsigned int num_hidden_layers;

    // Methods
    neural_network(unsigned int num_hidden_layers) : num_hidden_layers{num_hidden_layers} {
        layers_m = std::vector<matrix>(num_hidden_layers + 1);
        buf_m = std::vector<gradient_data>(num_hidden_layers + 2);

        buf_m[0].vals = (vector) new double[INPUT_SIZE];
        buf_m[0].deriv = (vector) new double[INPUT_SIZE];
        layers_m[0] = (matrix) new double[HIDDEN_SIZE * INPUT_SIZE];
        for (int i = 1; i < num_hidden_layers; ++i) {
            layers_m[i] = (matrix) new double[HIDDEN_SIZE * HIDDEN_SIZE];
            buf_m[i].vals = (vector) new double[HIDDEN_SIZE];
            buf_m[i].deriv = (vector) new double[HIDDEN_SIZE];
        }
        layers_m[num_hidden_layers] = (matrix) new double[OUTPUT_SIZE * HIDDEN_SIZE];
        buf_m[num_hidden_layers].vals = (vector) new double[HIDDEN_SIZE];
        buf_m[num_hidden_layers].deriv = (vector) new double[HIDDEN_SIZE];
        buf_m[num_hidden_layers + 1].vals = (vector) new double[OUTPUT_SIZE];
        buf_m[num_hidden_layers + 1].deriv = (vector) new double[OUTPUT_SIZE];

        input_m = layers_m[0];
        output_m = layers_m[num_hidden_layers + 1];
    }

    void propagate_forwards();
    void propagate_backwards();
    // DEBUG
    void dbg_print();
};

#endif /* ifndef NEURAL_NETWORK_H */

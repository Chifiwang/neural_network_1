#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "linalg.h"
#include "matrix.h"
#include <iostream>
#include <vector>

class neural_network {
public:
    // int inodes;
    // int hnodes;
    // int onodes;
    // double lr;

    // matrix wih;
    // matrix who;
    double learning_rate;
    int num_layers;
    std::vector<matrix> layers;
    std::vector<matrix> errors;
    std::vector<matrix> weights;


    matrix forward_propagate(matrix &in);
    void backward_propagate(matrix &targets);

    neural_network(int num_layers, int nodes_per_layer[],
                 double learning_rate)
      : num_layers{num_layers}, learning_rate{learning_rate} {

        layers = std::vector<matrix>(num_layers);
        errors = std::vector<matrix>(num_layers);
        weights = std::vector<matrix>(num_layers);

        for (int l = 0; l < num_layers - 1; ++l) {
            layers[l] = zero_matrix(nodes_per_layer[l], 1);
            errors[l] = zero_matrix(nodes_per_layer[l], 1);
            weights[l] = rand_matrix(nodes_per_layer[l+1], nodes_per_layer[l]);
        }

        layers[num_layers - 1] = zero_matrix(nodes_per_layer[num_layers - 1], 1);
        errors[num_layers - 1] = zero_matrix(nodes_per_layer[num_layers - 1], 1);

    }

    matrix query(matrix &input);
    void train(std::vector<std::pair<matrix, matrix>> &data, int epochs);
};

#endif /* ifndef NEURAL_NETWORK_H */

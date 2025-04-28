#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "utils.hpp"
#include <vector>

class neural_network {
public:
    double learning_rate;
    double momentum;
    int num_layers;
    std::vector<vector> layers;
    std::vector<vector> errors;
    std::vector<matrix> weights;


    vector forward_propagate(vector &in);
    void backward_propagate(vector &target);

    neural_network(int num_layers, int nodes_per_layer[],
                 double learning_rate, double momentum)
      : learning_rate{learning_rate}, momentum{momentum}, num_layers{num_layers} {

        layers = std::vector<vector>(num_layers);
        errors = std::vector<vector>(num_layers);
        weights = std::vector<matrix>(num_layers);

        for (int l = 0; l < num_layers - 1; ++l) {
            layers[l] = zero_vector(nodes_per_layer[l]);
            errors[l] = zero_vector(nodes_per_layer[l]);
            weights[l] = rand_matrix(nodes_per_layer[l], nodes_per_layer[l+1]);
        }

        layers[num_layers - 1] = zero_vector(nodes_per_layer[num_layers - 1]);
        errors[num_layers - 1] = zero_vector(nodes_per_layer[num_layers - 1]);

    }

    vector query(vector &input);
    void train(std::vector<std::pair<vector, vector>> &data, int epochs);
};

#endif /* ifndef NEURAL_NETWORK_H */

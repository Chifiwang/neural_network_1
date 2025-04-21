#include "neural_network.h"
#include "matrix.h"
#include <algorithm>
#include <cassert>
#include <random>
#include <utility>

using std::vector;

matrix neural_network::forward_propagate(matrix &inputs) {
    // layers[0] = inputs;
    for (int i = 0; i < inputs.r; i++) {
        layers[0].mat[i] = inputs.mat[i];
    }
    for (int l = 1; l < num_layers; l++) {
        for (int i = 0; i < layers[l-1].r; i++) {
            for (int j = 0; j < layers[l].r; j++) {
                // GOD MODE:
                // layers[l].mat[i] += weights[l-1].mat[i * weights[l-1].c + j] * layers[l-1].mat[j];
                layers[l].mat[j] += weights[l-1].mat[j * weights[l-1].c + i] * layers[l-1].mat[i];
                // printf("i: %d, %d\n", j, i);
            }
        }
        for (int i = 0; i < layers[l].r; i++) {
            layers[l].mat[i] = sigmoid(layers[l].mat[i]);
        }
    }

    return layers[num_layers - 1];
}

void neural_network::backward_propagate(matrix &targets) {
    // printf("%d %d %d\n", targets.r, layers[num_layers -1].r, errors[num_layers -1 ].r);
    for (int i = 0; i < layers[num_layers - 1].r; ++i) {
        errors[num_layers - 1].mat[i] = targets.mat[i] - layers[num_layers - 1].mat[i];
    }

    for (int l = num_layers - 1; l > 0; l--) {
        for (int i = 0; i < layers[l-1].r; i++) {
            errors[l-1].mat[i] = 0;
            for (int j = 0; j < layers[l].r; j++) {
                errors[l-1].mat[i] += weights[l-1].mat[i * weights[l-1].c + j] * errors[l].mat[j];
            }
            for (int j = 0; j < layers[l].r; j++) {
                weights[l-1].mat[i * weights[l-1].c + j] += learning_rate * errors[l].mat[j]
                    * layers[l].mat[j] * (1 - layers[l].mat[j]) * layers[l-1].mat[i];
            }
        }
    }
}

matrix neural_network::query(matrix &inputs) {
    return forward_propagate(inputs);
}

void neural_network::train(std::vector<std::pair<matrix, matrix>> &data, int epochs = 10) {
    printf("%d %d %d\n", weights[0].c, weights[1].c, weights[2].c);
    for (int e = 0; e < epochs; ++e) {
        std::shuffle(data.begin(), data.end(), std::random_device());
        for (int n = 0; n < data.size(); ++n) {
            forward_propagate(data[n].first);
            backward_propagate(data[n].second);
        }
    }
}

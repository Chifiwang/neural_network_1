#include "neural_network.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cassert>
#include <random>
#include <utility>

vector neural_network::forward_propagate(vector &input) {
    for (int i = 0; i < layers[0].n; ++i) {
        layers[0][i] = input[i];
    }

    for (int l = 1; l < num_layers; ++l) {
        for (int r = 0; r < layers[l-1].n; ++r) {
            for (int c = 0; c < layers[l].n; ++c) {
                layers[l][c] += weights[l-1][r][c] * layers[l-1][r];
            }
        }
        for (int i = 0; i < layers[l].n; ++i) {
            layers[l][i] = sigmoid(layers[l][i]);
        }
    }

    return layers[num_layers-1];
}

void neural_network::backward_propagate(vector &targets) {
    for (int i = 0; i < targets.n; ++i) {
        errors[num_layers-1][i] = targets[i] - layers[num_layers-1][i];
    }

    for (int l = num_layers-1; l > 0; --l) {
        for (int c = 0; c < layers[l-1].n; ++c) {
            errors[l-1][c] = 0;
            for (int r = 0; r < layers[l].n; ++r) {
                errors[l-1][c] += weights[l-1][c][r] * errors[l][r];
            }
            for (int r = 0; r < layers[l].n; ++r) {
                weights[l-1][c][r] += learning_rate
                    * layers[l][r] * (1 - layers[l][r])
                    * errors[l][r] * layers[l-1][c];
            }
        }
    }
}

vector neural_network::query(vector &input) {
    return forward_propagate(input);
}

void neural_network::train(std::vector<std::pair<vector, vector>> &data, int epochs = 10) {
    // printf("%d %d %d\n", weights[0].c, weights[1].c, weights[2].c);
    // const long max = epochs * data.size();
    for (int e = 0; e < epochs; ++e) {
        // std::shuffle(data.begin(), data.end(), std::random_device());
        printf("\r Epoch %d / %d completed", e, epochs);
        for (std::size_t n = 0; n < data.size(); ++n) {
            forward_propagate(data[n].first);
            backward_propagate(data[n].second);
            // printf("\r %ld / %ld completed", e * data.size() + n, max);
        }
        learning_rate *= momentum;
    }

    printf("\n");
}

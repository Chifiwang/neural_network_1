#include "utils.hpp"
#include "neural_network.hpp"
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#define TRAINING_SET "mnist_mini_train.csv"
#define TESTING_SET "mnist_mini_test.csv"
#define FULL_TRAIN "mnist_train.csv"
#define FULL_TEST "mnist_test.csv"

void init_data(std::string name, std::vector<vector> &inputs, std::vector<vector> &targets) {
    std::ifstream train;
    train.open(name);

    int target;
    std::string in;
    int c = 0;
    while (std::getline(train, in, '\n')) {
        std::istringstream iss(in);
        std::string token;

        std::getline(iss, token, ',');
        target = std::stoi(token);

        inputs.push_back(zero_vector(784));
        targets.push_back(zero_vector(10));

        for (int i = 0; i < 10; ++i) {
            if (i == target) {
                targets[c][i] = 1;
            } else {
                targets[c][i] = 0;
            }
        }

        int r = 0;
        while (std::getline(iss, token, ',')) {
            inputs[c][r] = std::stoi(token) / 255.0;
            r++;
        }

        c++;
    }
}

int main() {
    int layer_sizes[] = {784, 500, 10};
    neural_network n = neural_network(3, layer_sizes, 0.17, 0.85);

    std::vector<vector> training_inputs = std::vector<vector>();
    std::vector<vector> training_targets = std::vector<vector>();

    init_data(FULL_TRAIN, training_inputs, training_targets);

    std::vector<std::pair<vector, vector>> data = std::vector<std::pair<vector, vector>>(training_targets.size());
    for (int i = 0; i < training_inputs.size(); ++i) {
        data[i] = std::make_pair(training_inputs[i], training_targets[i]);
    }

    n.train(data, 10);

    std::vector<vector> testing_inputs = std::vector<vector>();
    std::vector<vector> testing_targets = std::vector<vector>();


    init_data(FULL_TEST, testing_inputs, testing_targets);

    double acc = 0;
    for (std::size_t in = 0; in < testing_inputs.size(); ++in) {
        vector o = n.query(testing_inputs[in]);
        int mx = 0, t = 0;
        double mx_v = -100, mn = 0;
        for (int i = 0; i < testing_targets[in].n; ++i) {
            if (testing_targets[in][i] > 0) {
                t = i;
                break;
            }
        }
        for (int j = 0; j < o.n; ++j) {
            if (mx_v < o[j]) {
                mx = j;
                mx_v = o[j];
            }
            printf("%f ", o[j]);
        }


        printf("%d %d %f\n", mx, t, mx_v);
        acc += mx == t;
    }

    printf("%f\n", acc / testing_inputs.size());
}

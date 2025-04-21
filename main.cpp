#include "matrix.h"
#include "neural_network.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#define TRAINING_SET "mnist_mini_train.csv"
#define TESTING_SET "mnist_mini_test.csv"
#define FULL_TRAIN "mnist_train.csv"
#define FULL_TEST "mnist_test.csv"
#define TEST_TRAINING "test_train.csv"

void init_data(std::string name, std::vector<matrix> &inputs, std::vector<matrix> &targets) {
    std::ifstream train;
    train.open(name);

    int target;
    std::string in;
    int c = 0;
    while (std::getline(train, in)) {
        std::istringstream iss(in);
        std::string token;

        std::getline(iss, token, ',');
        target = std::stoi(token);

        inputs.push_back(zero_matrix(784, 1));
        targets.push_back(zero_matrix(10, 1));

        for (int i = 0; i < 10; ++i) {
            if (i == target) {
                targets[c].mat[i] = 1;
            } else {
                targets[c].mat[i] = 0;
            }
        }

        int r = 0;
        while (std::getline(iss, token, ',')) {
            inputs[c].mat[r] = std::stoi(token) / 255.0;
            r++;
        }
        c++;
    }
}

int main() {
    int layer_sizes[] = {784, 500, 10};
    // int layer_sizes[] = {3, 4, 3};
    neural_network n = neural_network(3, layer_sizes, 0.17);

    // std::vector<matrix> training_inputs = std::vector<matrix>(1);
    // std::vector<matrix> training_targets = std::vector<matrix>(1);
    std::vector<matrix> training_inputs = std::vector<matrix>();
    std::vector<matrix> training_targets = std::vector<matrix>();

    init_data(TRAINING_SET, training_inputs, training_targets);

    std::vector<std::pair<matrix, matrix>> data = std::vector<std::pair<matrix, matrix>>(training_targets.size());
    for (int i = 0; i < training_inputs.size(); ++i) {
        data[i] = std::make_pair(training_inputs[i], training_targets[i]);
    }

    n.train(data, 10);
    // // n.query(training_inputs[0]);
    //
    // print_matrix(n.weights[0]);
    // printf("\n");
    // print_matrix(n.weights[1]);
    // printf("\n");
    // // print_matrix(n.layers[2]);
    //
    std::vector<matrix> testing_inputs = std::vector<matrix>();
    std::vector<matrix> testing_targets = std::vector<matrix>();


    init_data(TESTING_SET, testing_inputs, testing_targets);

    double acc = 0;
    for (int in = 0; in < testing_inputs.size(); ++in) {
        matrix o = n.query(testing_targets[in]);
        for (int i = 0; i < o.c; ++i) {
            int mx = 0, t = 0;
            double mx_v = -100, mn = 0;
            for (int j = 0; j < o.r; ++j) {
                if (at(testing_targets[in], j, i) > 0.1) {
                    t = j;
                }
                if (mx_v < at(o, j, i)) {
                    mx = j;
                    mx_v = at(o, j, i);
                }
                printf("%f ", at(o, j, i));
            }

            printf("%d %d %f\n", mx, t, mx_v);
            acc += mx == t;
        }
    }

    printf("%f\n", acc / testing_inputs.size());
}

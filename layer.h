#ifndef LAYER_H
#define LAYER_H

#include <cstddef>
#define INPUT_SIZE 5
#define HIDDEN_SIZE 5
#define OUTPUT_SIZE 10

typedef double input_layer[INPUT_SIZE];
typedef double hidden_layer[HIDDEN_SIZE];
typedef double output_layer[OUTPUT_SIZE];

void propagate_forwards(double in[], double out[], std::size_t in_size, std::size_t out_size);

#endif /* ifndef LAYER_H */

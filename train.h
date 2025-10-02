#ifndef TRAIN_H
#define TRAIN_H

#include "layers.h"
#include <vector>

float cross_entropy_loss(const std::vector<float>& probs, int label);
std::vector<float> cross_entropy_grad(const std::vector<float>& probs, int label);

std::vector<float> flatten(const std::vector<std::vector<float>>& input2D);
std::vector<std::vector<float>> flatten_backward(const std::vector<float>& grad1D, const std::vector<std::vector<float>>& ref2D);

void train_epoch(
    std::vector<Layer*>& layers,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& images, // [N][C][H][W]
    const std::vector<int>& labels,
    float lr
);

void evaluate(
    const std::vector<Layer*>& layers,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& images, // [N][C][H][W]
    const std::vector<int>& labels
);

int predict(
    const std::vector<Layer*>& layers,
    const std::vector<std::vector<std::vector<float>>>& image3D // [C][H][W]
);

#endif

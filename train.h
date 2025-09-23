#ifndef TRAIN_H
#define TRAIN_H

#include "layers.h"
#include <vector>

float cross_entropy_loss(const std::vector<float>& probs, int label);
std::vector<float> cross_entropy_grad(const std::vector<float>& probs, int label);

std::vector<float> flatten(const std::vector<std::vector<float>>& input2D);
std::vector<std::vector<float>> flatten_backward(const std::vector<float>& grad1D, const std::vector<std::vector<float>>& ref2D);

// ======================== train / eval ========================
void train_epoch(
    Conv2D& conv,
    ReLU& relu,
    FullyConnected& fc,
    Softmax& softmax,
    const std::vector<std::vector<std::vector<float>>>& images, // [N][28][28]
    const std::vector<int>& labels,
    float lr
);

float evaluate(
    Conv2D& conv,
    ReLU& relu,
    FullyConnected& fc,
    Softmax& softmax,
    const std::vector<std::vector<std::vector<float>>>& images,
    const std::vector<int>& labels
);

#endif

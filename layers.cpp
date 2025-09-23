#include "layers.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// =================== Conv2D ===================
Conv2D::Conv2D(int input_dim, int kernel_size, int num_filters)
    : input_dim(input_dim), kernel_size(kernel_size), num_filters(num_filters) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1, 0.1);

    kernels.resize(num_filters, std::vector<std::vector<float>>(kernel_size, std::vector<float>(kernel_size)));
    for (int f = 0; f < num_filters; f++)
        for (int i = 0; i < kernel_size; i++)
            for (int j = 0; j < kernel_size; j++)
                kernels[f][i][j] = dist(gen);
}

std::vector<std::vector<float>> Conv2D::forward(const std::vector<std::vector<float>>& input) {
    last_input = input;
    // TODO: implementar convolução real
    return input; // placeholder
}

std::vector<std::vector<float>> Conv2D::backward(const std::vector<std::vector<float>>& grad_output) {
    // TODO: implementar backward
    return grad_output; // placeholder
}

void Conv2D::update(float lr) {
    // TODO: atualizar kernels
}

void Conv2D::debugPrint() const {
    std::cout << name() << " kernels:\n";
    for (int f = 0; f < num_filters; f++) {
        std::cout << "Filter " << f << ":\n";
        for (auto& row : kernels[f]) {
            for (auto& v : row) std::cout << std::fixed << std::setprecision(3) << v << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

// =================== ReLU ===================
std::vector<std::vector<float>> ReLU::forward(const std::vector<std::vector<float>>& input) {
    last_input = input;
    std::vector<std::vector<float>> out = input;
    for (auto& row : out)
        for (auto& v : row)
            v = std::max(0.0f, v);
    return out;
}

std::vector<std::vector<float>> ReLU::backward(const std::vector<std::vector<float>>& grad_output) {
    std::vector<std::vector<float>> grad = grad_output;
    for (size_t i = 0; i < grad.size(); i++)
        for (size_t j = 0; j < grad[i].size(); j++)
            grad[i][j] *= (last_input[i][j] > 0 ? 1.0f : 0.0f);
    return grad;
}

void ReLU::debugPrint() const {
    std::cout << name() << " last_input:\n";
    for (auto& row : last_input) {
        for (auto& v : row) std::cout << v << " ";
        std::cout << "\n";
    }
}

// =================== MaxPool2x2 ===================
MaxPool2x2::MaxPool2x2(int input_dim) : input_dim(input_dim) {}

std::vector<std::vector<float>> MaxPool2x2::forward(const std::vector<std::vector<float>>& input) {
    last_input = input;
    // TODO: implementar pooling real
    return input; // placeholder
}

std::vector<std::vector<float>> MaxPool2x2::backward(const std::vector<std::vector<float>>& grad_output) {
    // TODO: implementar backward
    return grad_output; // placeholder
}

void MaxPool2x2::debugPrint() const {
    std::cout << name() << " last_input:\n";
    for (auto& row : last_input) {
        for (auto& v : row) std::cout << v << " ";
        std::cout << "\n";
    }
}

// =================== FullyConnected ===================
FullyConnected::FullyConnected(int in_size, int out_size)
    : in_size(in_size), out_size(out_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1, 0.1);

    weights.resize(out_size, std::vector<float>(in_size));
    bias.resize(out_size);
    for (int i = 0; i < out_size; i++) {
        bias[i] = dist(gen);
        for (int j = 0; j < in_size; j++)
            weights[i][j] = dist(gen);
    }
}

std::vector<float> FullyConnected::forward(const std::vector<float>& input) {
    last_input = input;
    std::vector<float> out(out_size, 0.0f);
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++)
            out[i] += weights[i][j] * input[j];
        out[i] += bias[i];
    }
    return out;
}

std::vector<float> FullyConnected::backward(const std::vector<float>& grad_output) {
    std::vector<float> grad_input(in_size, 0.0f);
    for (int i = 0; i < out_size; i++)
        for (int j = 0; j < in_size; j++)
            grad_input[j] += weights[i][j] * grad_output[i];
    return grad_input;
}

void FullyConnected::update(float lr) {
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++)
            weights[i][j] -= lr * last_input[j]; // placeholder, precisa gradiente real
        bias[i] -= lr; // placeholder
    }
}

void FullyConnected::debugPrint() const {
    std::cout << name() << " weights:\n";
    for (auto& row : weights) {
        for (auto& v : row) std::cout << v << " ";
        std::cout << "\n";
    }
}

// =================== Softmax ===================
std::vector<float> Softmax::forward(const std::vector<float>& input) {
    last_output = input;
    float max_val = *std::max_element(input.begin(), input.end());
    std::vector<float> out(input.size());
    float sum = 0;
    for (size_t i = 0; i < input.size(); i++) {
        out[i] = std::exp(input[i] - max_val);
        sum += out[i];
    }
    for (auto& v : out) v /= sum;
    last_output = out;
    return out;
}

std::vector<float> Softmax::backward(const std::vector<float>& grad_output) {
    // TODO: cross-entropy simplifica para gradiente
    std::vector<float> grad(grad_output.size());
    for (size_t i = 0; i < grad.size(); i++)
        grad[i] = grad_output[i]; // placeholder
    return grad;
}

void Softmax::debugPrint() const {
    std::cout << "Softmax output n precisa dessa merda mas a virtual layer n deixa tirar af: ";
    for (float x : last_output) std::cout << x << " ";
    std::cout << "\n";
}
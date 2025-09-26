#include "layers.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <omp.h>

// =================== Conv2D ===================

Conv2D::Conv2D(int input_dim, int kernel_size, int num_filters)
    : input_dim(input_dim), kernel_size(kernel_size), num_filters(num_filters) 
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1, 0.1);

    kernels.resize(num_filters);
    grad_kernels.resize(num_filters);

    for (int f = 0; f < num_filters; f++) {
        kernels[f].resize(kernel_size, std::vector<float>(kernel_size));
        grad_kernels[f].resize(kernel_size, std::vector<float>(kernel_size, 0.0f));

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                kernels[f][i][j] = dist(gen);
            }
        }
    }
}

double Conv2D::total_forward_time = 0.0;
double Conv2D::total_backward_time = 0.0;
double Conv2D::total_update_time = 0.0;

std::vector<std::vector<std::vector<float>>> Conv2D::forward(const std::vector<std::vector<float>>& input) {

    auto start = std::chrono::high_resolution_clock::now();

    last_input = input;
    int out_dim = input_dim - kernel_size + 1;
    std::vector<std::vector<std::vector<float>>> output(
        num_filters, std::vector<std::vector<float>>(out_dim, std::vector<float>(out_dim, 0.0f))
    );

    // parallelize over filters and output positions
    #pragma omp parallel for collapse(3) schedule(static)
    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < out_dim; i++) {
            for (int j = 0; j < out_dim; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < kernel_size; ki++) {
                    for (int kj = 0; kj < kernel_size; kj++) {
                        sum += input[i + ki][j + kj] * kernels[f][ki][kj];
                    }
                }
                output[f][i][j] = sum;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return output;
}

std::vector<std::vector<float>> Conv2D::backward(const std::vector<std::vector<std::vector<float>>>& grad_output) {

    auto start = std::chrono::high_resolution_clock::now();

    int out_dim = grad_output[0].size();
    int in_dim = input_dim;

    // gradiente no input
    std::vector<std::vector<float>> grad_input(in_dim, std::vector<float>(in_dim, 0.0f));

    // zera grad_kernels
    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                grad_kernels[f][i][j] = 0.0f;
            }
        }
    }

    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < out_dim; i++) {
            for (int j = 0; j < out_dim; j++) {
                float go = grad_output[f][i][j];
                for (int ki = 0; ki < kernel_size; ki++) {
                    for (int kj = 0; kj < kernel_size; kj++) {
                        grad_kernels[f][ki][kj] += last_input[i + ki][j + kj] * go;
                        grad_input[i + ki][j + kj] += kernels[f][ki][kj] * go;
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return grad_input;
}

void Conv2D::update(float lr) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                kernels[f][i][j] -= lr * grad_kernels[f][i][j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_update_time += std::chrono::duration<double, std::milli>(end - start).count();
}

void Conv2D::debugPrint() const {
    std::cout << "Conv2D with " << num_filters << " filters (" << kernel_size << "x" << kernel_size << ")\n";
}

// =================== ReLU ===================
double ReLU::total_forward_time = 0.0;
double ReLU::total_backward_time = 0.0;
std::vector<std::vector<float>> ReLU::forward(const std::vector<std::vector<float>>& input) {

    auto start = std::chrono::high_resolution_clock::now();

    last_input = input;
    std::vector<std::vector<float>> out = input;
    for (auto& row : out)
        for (auto& v : row)
            v = std::max(0.0f, v);

    auto end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();    

    return out;
}

std::vector<std::vector<float>> ReLU::backward(const std::vector<std::vector<float>>& grad_output) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<float>> grad = grad_output;
    for (size_t i = 0; i < grad.size(); i++)
        for (size_t j = 0; j < grad[i].size(); j++)
            grad[i][j] *= (last_input[i][j] > 0 ? 1.0f : 0.0f);

    auto end = std::chrono::high_resolution_clock::now();
    total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();
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
double MaxPool2x2::total_forward_time = 0.0;
double MaxPool2x2::total_backward_time = 0.0;

MaxPool2x2::MaxPool2x2(int input_dim) : input_dim(input_dim) {}

std::vector<std::vector<float>> MaxPool2x2::forward(const std::vector<std::vector<float>>& input) {

    auto start = std::chrono::high_resolution_clock::now();

    last_input = input;
    int out_dim = input_dim / 2;
    std::vector<std::vector<float>> output(out_dim, std::vector<float>(out_dim));

    max_indices.clear();
    max_indices.reserve(out_dim * out_dim);

    // parallelize outer loops; each output cell independent
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            float max_val = -1e9;
            int max_idx = -1;

            // percorre janela 2x2
            for (int di = 0; di < 2; di++) {
                for (int dj = 0; dj < 2; dj++) {
                    int r = i * 2 + di;
                    int c = j * 2 + dj;
                    if (input[r][c] > max_val) {
                        max_val = input[r][c];
                        max_idx = r * input_dim + c; // index linearizado
                    }
                }
            }
            output[i][j] = max_val;
            max_indices.push_back(max_idx);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return output;
}

std::vector<std::vector<float>> MaxPool2x2::backward(const std::vector<std::vector<float>>& grad_output) {

    auto start = std::chrono::high_resolution_clock::now();

    int out_dim = grad_output.size();
    std::vector<std::vector<float>> grad_input(input_dim, std::vector<float>(input_dim, 0.0f));

    // each output contributes to a single input index -> independent writes
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            int idx = i * out_dim + j;
            int max_idx = max_indices[idx];
            int r = max_idx / input_dim;
            int c = max_idx % input_dim;
            grad_input[r][c] = grad_output[i][j];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return grad_input;
}

void MaxPool2x2::debugPrint() const {
    std::cout << name() << " last_input:\n";
    for (auto& row : last_input) {
        for (auto& v : row) std::cout << std::setw(6) << v << " ";
        std::cout << "\n";
    }
}

// =================== FullyConnected ===================
double FullyConnected::total_forward_time = 0.0;
double FullyConnected::total_backward_time = 0.0;
double FullyConnected::total_update_time = 0.0;

FullyConnected::FullyConnected(int in_size, int out_size)
    : in_size(in_size), out_size(out_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1, 0.1);

    weights.resize(out_size, std::vector<float>(in_size));
    bias.resize(out_size);
    grad_weights.resize(out_size, std::vector<float>(in_size, 0.0f));
    grad_bias.resize(out_size, 0.0f);

    for (int i = 0; i < out_size; i++) {
        bias[i] = dist(gen);
        for (int j = 0; j < in_size; j++)
            weights[i][j] = dist(gen);
    }
}

std::vector<float> FullyConnected::forward(const std::vector<float>& input) {

    auto start = std::chrono::high_resolution_clock::now();

    last_input = input;
    std::vector<float> out(out_size, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++) {
            out[i] += weights[i][j] * input[j];
        }
        out[i] += bias[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return out;
}

std::vector<float> FullyConnected::backward(const std::vector<float>& grad_output) {

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> grad_input(in_size, 0.0f);

    for (int i = 0; i < out_size; i++) {
        grad_bias[i] = 0.0f;
        for (int j = 0; j < in_size; j++)
            grad_weights[i][j] = 0.0f;
    }

    for (int i = 0; i < out_size; i++) {
        grad_bias[i] += grad_output[i];   // dL/db = grad_output
        for (int j = 0; j < in_size; j++) {
            grad_weights[i][j] += last_input[j] * grad_output[i]; // dL/dW
            grad_input[j] += weights[i][j] * grad_output[i];      // dL/dInput
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return grad_input;
}

void FullyConnected::update(float lr) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++) {
            weights[i][j] -= lr * grad_weights[i][j];
        }
        bias[i] -= lr * grad_bias[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    total_update_time += std::chrono::duration<double, std::milli>(end - start).count();
}

void FullyConnected::debugPrint() const {
    std::cout << name() << " weights:\n";
    for (int i = 0; i < std::min(out_size, 3); i++) {
        for (int j = 0; j < std::min(in_size, 5); j++) {
            std::cout << std::fixed << std::setprecision(4) << weights[i][j] << " ";
        }
        std::cout << "... bias=" << bias[i] << "\n";
    }
}
// =================== Softmax ===================
double Softmax::total_forward_time = 0.0;
double Softmax::total_backward_time = 0.0;
std::vector<float> Softmax::forward(const std::vector<float>& input) {

    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<float> Softmax::backward(const std::vector<float>& grad_output) { // ta usando cross-entropy ent n precisa
    return grad_output;
}

void Softmax::debugPrint() const {
    std::cout << "Softmax output n precisa dessa merda mas a virtual layer n deixa tirar af: ";
    for (float x : last_output) std::cout << x << " ";
    std::cout << "\n";
}
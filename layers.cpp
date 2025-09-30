#include "layers.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>

// ================================================================== CONV2D

double Conv2D::total_forward_time = 0.0;
double Conv2D::total_backward_time = 0.0;

Conv2D::Conv2D(int in_c, int out_c, int k, int s)
    : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s) {

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    weights.resize(out_c * in_c * k * k);
    for (auto &w : weights) w = dist(gen);

    bias.resize(out_c, 0.0f);

    grad_weights = Tensor(std::vector<float>(weights.size(), 0.0f),
                          {out_c, in_c, k, k});
    grad_bias.resize(out_c, 0.0f);
}

std::string Conv2D::name() const { return "Conv2D"; }

// Tensor Conv2D::forward(const Tensor& input) {

//     auto start = std::chrono::high_resolution_clock::now();

//     input_cache = input;

//     int H = input.shape[1];
//     int W = input.shape[2];
//     int out_H = (H - kernel_size) / stride + 1;
//     int out_W = (W - kernel_size) / stride + 1;

//     Tensor out(std::vector<float>(out_channels * out_H * out_W, 0.0f),
//                {out_channels, out_H, out_W});

//     for (int oc = 0; oc < out_channels; oc++) {
//         for (int i = 0; i < out_H; i++) {
//             for (int j = 0; j < out_W; j++) {
//                 float sum = bias[oc];
//                 for (int ic = 0; ic < in_channels; ic++) {
//                     for (int ki = 0; ki < kernel_size; ki++) {
//                         for (int kj = 0; kj < kernel_size; kj++) {
//                             int xi = i * stride + ki;
//                             int xj = j * stride + kj;
//                             float w = weights[ ((oc*in_channels+ic)*kernel_size+ki)*kernel_size + kj ];
//                             sum += input.at(ic, xi, xj) * w;
//                         }
//                     }
//                 }
//                 out.at(oc, i, j) = sum;
//             }
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

//     return out;
// }

// Tensor Conv2D::backward(const Tensor& grad_output) {

//     auto start = std::chrono::high_resolution_clock::now();

//     int H = input_cache.shape[1];
//     int W = input_cache.shape[2];
//     int out_H = grad_output.shape[1];
//     int out_W = grad_output.shape[2];

//     Tensor grad_input(std::vector<float>(in_channels * H * W, 0.0f),
//                       {in_channels, H, W});

//     std::fill(grad_weights.data.begin(), grad_weights.data.end(), 0.0f);
//     std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);

//     for (int oc = 0; oc < out_channels; oc++) {
//         for (int i = 0; i < out_H; i++) {
//             for (int j = 0; j < out_W; j++) {
//                 float go = grad_output.at(oc, i, j);
//                 grad_bias[oc] += go;

//                 for (int ic = 0; ic < in_channels; ic++) {
//                     for (int ki = 0; ki < kernel_size; ki++) {
//                         for (int kj = 0; kj < kernel_size; kj++) {
//                             int xi = i * stride + ki;
//                             int xj = j * stride + kj;

//                             grad_weights.data[ ((oc*in_channels+ic)*kernel_size+ki)*kernel_size + kj ]
//                                 += input_cache.at(ic, xi, xj) * go;

//                             grad_input.at(ic, xi, xj) +=
//                                 weights[ ((oc*in_channels+ic)*kernel_size+ki)*kernel_size + kj ] * go;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();

//     return grad_input;
// }

Tensor Conv2D::forward(const Tensor& input) {
    input_cache = input;

    int N = input.shape[0];
    int H = input.shape[2];
    int W = input.shape[3];
    int out_H = (H - kernel_size) / stride + 1;
    int out_W = (W - kernel_size) / stride + 1;

    Tensor out(std::vector<float>(N * out_channels * out_H * out_W, 0.0f),
               {N, out_channels, out_H, out_W});

    // #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int i = 0; i < out_H; i++) {
                for (int j = 0; j < out_W; j++) {
                    float sum = bias[oc];
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int ki = 0; ki < kernel_size; ki++) {
                            for (int kj = 0; kj < kernel_size; kj++) {
                                int xi = i * stride + ki;
                                int xj = j * stride + kj;
                                float w = weights[ ((oc*in_channels+ic)*kernel_size+ki)*kernel_size + kj ];
                                sum += input.at(n, ic, xi, xj) * w;
                            }
                        }
                    }
                    out.at(n, oc, i, j) = sum;
                }
            }
        }
    }
    return out;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    int N = grad_output.shape[0];
    int H = input_cache.shape[2];
    int W = input_cache.shape[3];
    int out_H = grad_output.shape[2];
    int out_W = grad_output.shape[3];

    Tensor grad_input(std::vector<float>(N * in_channels * H * W, 0.0f),
                      {N, in_channels, H, W});

    std::fill(grad_weights.data.begin(), grad_weights.data.end(), 0.0f);
    std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);

    // #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int i = 0; i < out_H; i++) {
                for (int j = 0; j < out_W; j++) {
                    float go = grad_output.at(n, oc, i, j);
                    // #pragma omp atomic
                    grad_bias[oc] += go;

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int ki = 0; ki < kernel_size; ki++) {
                            for (int kj = 0; kj < kernel_size; kj++) {
                                int xi = i*stride + ki;
                                int xj = j*stride + kj;

                                // #pragma omp atomic
                                grad_weights.data[((oc*in_channels+ic)*kernel_size+ki)*kernel_size + kj] +=
                                    input_cache.at(n, ic, xi, xj) * go;

                                grad_input.at(n, ic, xi, xj) +=
                                    weights[((oc*in_channels+ic)*kernel_size+ki)*kernel_size + kj] * go;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

void Conv2D::update(float lr) {
    for (size_t idx = 0; idx < weights.size(); idx++) {
        weights[idx] -= lr * grad_weights.data[idx];
    }
    for (size_t oc = 0; oc < bias.size(); oc++) {
        bias[oc] -= lr * grad_bias[oc];
    }
}

// ================================================================== RELU

double ReLU::total_forward_time = 0.0;
double ReLU::total_backward_time = 0.0;

ReLU::ReLU() {}

std::string ReLU::name() const {
    return "ReLU";
}

// Tensor ReLU::forward(const Tensor& input) {
//     auto start = std::chrono::high_resolution_clock::now();

//     input_cache = input;
//     Tensor out(input.data, input.shape);

//     for (size_t idx = 0; idx < out.data.size(); idx++) {
//         out.data[idx] = std::max(0.0f, out.data[idx]);
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

//     return out;
// }

// Tensor ReLU::backward(const Tensor& grad_output) {
//     auto start = std::chrono::high_resolution_clock::now();

//     Tensor grad(grad_output.data, grad_output.shape);

//     for (int c = 0; c < grad.shape[0]; c++) {
//         for (int i = 0; i < grad.shape[1]; i++) {
//             for (int j = 0; j < grad.shape[2]; j++) {
//                 grad.at(c, i, j) *= (input_cache.at(c, i, j) > 0 ? 1.0f : 0.0f);
//             }
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();

//     return grad;
// }

Tensor ReLU::forward(const Tensor& input) {
    auto start = std::chrono::high_resolution_clock::now();

    input_cache = input;  // cache input for backward
    Tensor out(input.data, input.shape);

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int i = 0; i < H; i++)
                for (int j = 0; j < W; j++)
                    out.at(n, c, i, j) = std::max(0.0f, input.at(n, c, i, j));

    auto end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return out;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    auto start = std::chrono::high_resolution_clock::now();

    Tensor grad(grad_output.data, grad_output.shape);

    int N = grad.shape[0];
    int C = grad.shape[1];
    int H = grad.shape[2];
    int W = grad.shape[3];

    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int i = 0; i < H; i++)
                for (int j = 0; j < W; j++)
                    grad.at(n, c, i, j) *= (input_cache.at(n, c, i, j) > 0 ? 1.0f : 0.0f);

    auto end = std::chrono::high_resolution_clock::now();
    total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return grad;
}

// ================================================================== MAXPOOL

double MaxPool2x2::total_forward_time = 0.0;
double MaxPool2x2::total_backward_time = 0.0;

MaxPool2x2::MaxPool2x2() {}

std::string MaxPool2x2::name() const {
    return "MaxPool2x2";
}

// Tensor MaxPool2x2::forward(const Tensor& input) {
//     auto start = std::chrono::high_resolution_clock::now();

//     input_cache = input;
//     int C = input.shape[0];
//     int H = input.shape[1];
//     int W = input.shape[2];

//     int out_H = H / 2;
//     int out_W = W / 2;

//     Tensor output(std::vector<float>(C * out_H * out_W, 0.0f), {C, out_H, out_W});
//     max_indices.clear();
//     max_indices.reserve(C * out_H * out_W);

//     for (int c = 0; c < C; c++) {
//         for (int i = 0; i < out_H; i++) {
//             for (int j = 0; j < out_W; j++) {
//                 float max_val = -1e9;
//                 int max_idx = -1;

//                 for (int di = 0; di < 2; di++) {
//                     for (int dj = 0; dj < 2; dj++) {
//                         int r = i * 2 + di;
//                         int col = j * 2 + dj;
//                         float val = input.at(c, r, col);
//                         if (val > max_val) {
//                             max_val = val;
//                             max_idx = (c * H + r) * W + col; 
//                         }
//                     }
//                 }
//                 output.at(c, i, j) = max_val;
//                 max_indices.push_back(max_idx);
//             }
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

//     // std::cout << "MaxPool2x2 output shape: [" << C << ", " << out_H << ", " << out_W << "]\n";

//     return output;
// }

// Tensor MaxPool2x2::backward(const Tensor& grad_output) {
//     auto start = std::chrono::high_resolution_clock::now();

//     int C = input_cache.shape[0];
//     int H = input_cache.shape[1];
//     int W = input_cache.shape[2];

//     Tensor grad_input(std::vector<float>(C * H * W, 0.0f), {C, H, W});
//     int out_H = grad_output.shape[1];
//     int out_W = grad_output.shape[2];

//     for (int c = 0; c < C; c++) {
//         for (int i = 0; i < out_H; i++) {
//             for (int j = 0; j < out_W; j++) {
//                 int idx = (c * out_H + i) * out_W + j;
//                 int max_idx = max_indices[idx];

//                 int r = (max_idx / W) % H;
//                 int cc = max_idx % W;

//                 grad_input.at(c, r, cc) = grad_output.at(c, i, j);
//             }
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();

//     return grad_input;
// }

Tensor MaxPool2x2::forward(const Tensor& input) {
    input_cache = input;

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int out_H = H / 2;
    int out_W = W / 2;

    Tensor output(std::vector<float>(N*C*out_H*out_W, 0.0f), {N, C, out_H, out_W});
    max_indices.clear();
    max_indices.reserve(N*C*out_H*out_W);

    // #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < out_H; i++) {
                for (int j = 0; j < out_W; j++) {
                    float max_val = -1e9;
                    int max_idx = -1;
                    for (int di = 0; di < 2; di++) {
                        for (int dj = 0; dj < 2; dj++) {
                            int r = i*2 + di;
                            int col = j*2 + dj;
                            float val = input.at(n, c, r, col);
                            if (val > max_val) {
                                max_val = val;
                                max_idx = (c*H + r)*W + col;
                            }
                        }
                    }
                    output.at(n, c, i, j) = max_val;
                    // #pragma omp critical
                    max_indices.push_back(max_idx);
                }
            }
        }
    }
    return output;
}

Tensor MaxPool2x2::backward(const Tensor& grad_output) {
    int N = grad_output.shape[0];
    int C = grad_output.shape[1];
    int H = input_cache.shape[2];
    int W = input_cache.shape[3];
    int out_H = grad_output.shape[2];
    int out_W = grad_output.shape[3];

    Tensor grad_input(std::vector<float>(N*C*H*W, 0.0f), {N, C, H, W});

    // #pragma omp parallel for collapse(2)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < out_H; i++) {
                for (int j = 0; j < out_W; j++) {
                    int idx = (n*C + c) * out_H * out_W + i*out_W + j;
                    int max_idx = max_indices[idx];
                    int r = (max_idx / W) % H;
                    int cc = max_idx % W;
                    grad_input.at(n, c, r, cc) = grad_output.at(n, c, i, j);
                }
            }
        }
    }

    return grad_input;
}

// ================================================================== FULLYCONNECTED
double FullyConnected::total_forward_time = 0.0;
double FullyConnected::total_backward_time = 0.0;


FullyConnected::FullyConnected(int out_size) : out_size(out_size) {}

// void FullyConnected::init_from_tensor(const Tensor& input) {
//     in_size = 1;
//     for (auto dim : input.shape) in_size *= dim;

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

//     weights.resize(in_size * out_size);
//     bias.resize(out_size);
//     grad_weights.resize(in_size * out_size, 0.0f);
//     grad_bias.resize(out_size, 0.0f);

//     for (int i = 0; i < in_size * out_size; i++) weights[i] = dist(gen);
//     for (int i = 0; i < out_size; i++) bias[i] = dist(gen);
// }

void FullyConnected::init_from_tensor(const Tensor& input) {
    // input: [N, in_size, 1, 1] → we only want per-sample size
    in_size = input.shape[1] * input.shape[2] * input.shape[3];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    weights.resize(in_size * out_size);
    bias.resize(out_size);
    grad_weights.resize(in_size * out_size, 0.0f);
    grad_bias.resize(out_size, 0.0f);

    for (int i = 0; i < in_size * out_size; i++) weights[i] = dist(gen);
    for (int i = 0; i < out_size; i++) bias[i] = dist(gen);
}

std::string FullyConnected::name() const { return "FullyConnected"; }

// Tensor FullyConnected::forward(const Tensor& input) {

//     if (in_size == -1) init_from_tensor(input);

//     auto start = std::chrono::high_resolution_clock::now();

//     input_cache = input;
//     Tensor out(std::vector<float>(out_size, 0.0f), {out_size, 1, 1});

//     for (int i = 0; i < out_size; i++) {
//         float sum = bias[i];
//         for (int j = 0; j < in_size; j++)
//             sum += weights[i * in_size + j] * input.data[j];
//         out.data[i] = sum;
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

//     return out;
// }

// Tensor FullyConnected::backward(const Tensor& grad_output) {

//     auto start = std::chrono::high_resolution_clock::now();

//     Tensor grad_input(std::vector<float>(in_size, 0.0f), {in_size, 1, 1});

//     for (int i = 0; i < out_size; i++) {
//         grad_bias[i] = grad_output.data[i];
//         for (int j = 0; j < in_size; j++) {
//             grad_weights[i * in_size + j] = input_cache.data[j] * grad_output.data[i];
//             grad_input.data[j] += weights[i * in_size + j] * grad_output.data[i];
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_backward_time += std::chrono::duration<double, std::milli>(end - start).count();

//     return grad_input;
// }

// void FullyConnected::update(float lr) {
//     for (int i = 0; i < out_size; i++) {
//         for (int j = 0; j < in_size; j++)
//             weights[i * in_size + j] -= lr * grad_weights[i * in_size + j];
//         bias[i] -= lr * grad_bias[i];
//     }
// }

// Tensor FullyConnected::forward(const Tensor& input) {
//     // input.shape = [N, C, H, W]
//     if (in_size == -1) init_from_tensor(input);

//     auto start = std::chrono::high_resolution_clock::now();

//     int N = input.shape[0];
//     Tensor out(std::vector<float>(N * out_size, 0.0f), {N, out_size, 1, 1});
//     input_cache = input;

//     for (int n = 0; n < N; n++) {
//         for (int i = 0; i < out_size; i++) {
//             float sum = bias[i];
//             for (int j = 0; j < in_size; j++)
//                 sum += weights[i * in_size + j] * input.data[n * in_size + j]; // flattened input per batch
//             out.data[n * out_size + i] = sum;
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();
//     return out;
// }

// Tensor FullyConnected::backward(const Tensor& grad_output) {
//     int N = grad_output.shape[0];
//     Tensor grad_input(std::vector<float>(N * in_size, 0.0f), {N, in_size, 1, 1});

//     // zero grads
//     std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
//     std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);

//     for (int n = 0; n < N; n++) {
//         for (int i = 0; i < out_size; i++) {
//             grad_bias[i] += grad_output.data[n * out_size + i]; // accumulate over batch
//             for (int j = 0; j < in_size; j++) {
//                 grad_weights[i * in_size + j] += input_cache.data[n * in_size + j] * grad_output.data[n * out_size + i];
//                 grad_input.data[n * in_size + j] += weights[i * in_size + j] * grad_output.data[n * out_size + i];
//             }
//         }
//     }

//     return grad_input;
// }

// input: [N, in_size, 1, 1] → output: [N, out_size, 1, 1]
Tensor FullyConnected::forward(const Tensor& input) {
    if (in_size == -1) {
        in_size = input.shape[1]; // flattened size per sample
        init_from_tensor(input);
    }

    // std::cout << "FullyConnected input shape: [" << input.shape[0] << ", " << input.shape[1] << ", " << input.shape[2] << ", " << input.shape[3] << "]\n";

    int N = input.shape[0];
    Tensor out(std::vector<float>(N * out_size), {N, out_size, 1, 1});

    input_cache = input; // save for backward

    for (int n = 0; n < N; n++) {
        for (int i = 0; i < out_size; i++) {
            float sum = bias[i];
            for (int j = 0; j < in_size; j++) {
                sum += weights[i * in_size + j] * input.data[n*in_size + j];
            }
            out.data[n*out_size + i] = sum;
        }
    }

    return out;
}

Tensor FullyConnected::backward(const Tensor& grad_output) {
    int N = grad_output.shape[0];

    // reset gradients
    std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
    std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);

    Tensor grad_input(std::vector<float>(N * in_size, 0.0f), {N, in_size, 1, 1});

    // #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < out_size; i++) {
            float go = grad_output.data[n*out_size + i];
            // #pragma omp atomic
            grad_bias[i] += go;

            for (int j = 0; j < in_size; j++) {
                // #pragma omp atomic
                grad_weights[i*in_size + j] += input_cache.data[n*in_size + j] * go;

                grad_input.data[n*in_size + j] += weights[i*in_size + j] * go;
            }
        }
    }

    return grad_input;
}

void FullyConnected::update(float lr) {
    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++)
            weights[i * in_size + j] -= lr * grad_weights[i * in_size + j];
        bias[i] -= lr * grad_bias[i];
    }
}

// ================================================================== SOFTMAX
double Softmax::total_forward_time = 0.0;
double Softmax::total_backward_time = 0.0;

std::string Softmax::name() const { return "Softmax"; }

Tensor Softmax::forward(const Tensor& input) {
    // input.shape = [N, F, 1, 1]
    auto start = std::chrono::high_resolution_clock::now();

    int N = input.shape[0];
    int F = input.shape[1];  // number of features / classes
    Tensor out(std::vector<float>(N * F, 0.0f), {N, F, 1, 1});
    output_cache = out;  // store output for backward

    for (int n = 0; n < N; n++) {
        // find max for numerical stability
        float max_val = -1e9;
        for (int i = 0; i < F; i++)
            max_val = std::max(max_val, input.data[n * F + i]);

        float sum = 0.0f;
        for (int i = 0; i < F; i++) {
            out.data[n * F + i] = std::exp(input.data[n * F + i] - max_val);
            sum += out.data[n * F + i];
        }

        for (int i = 0; i < F; i++)
            out.data[n * F + i] /= sum;
    }

    output_cache = out;  // cache output for backward

    auto end = std::chrono::high_resolution_clock::now();
    total_forward_time += std::chrono::duration<double, std::milli>(end - start).count();

    return out;
}


Tensor Softmax::backward(const Tensor& grad_output) {
    return grad_output;
}

// ================================================================== FLATTEN

// [C,H,W] -> [C*H*W, 1, 1]
// Tensor Flatten::forward(const Tensor& input) {
//     input_shape = input.shape; // save [N, C, H, W]
//     int N = input.shape[0];
//     int C = input.shape[1];
//     int H = input.shape[2];
//     int W = input.shape[3];
//     int flat_size = C * H * W;

//     Tensor out(std::vector<float>(N * flat_size, 0.0f), {N, flat_size, 1, 1});

//     // #pragma omp parallel for
//     for (int n = 0; n < N; n++) {
//         for (int c = 0; c < C; c++) {
//             for (int h = 0; h < H; h++) {
//                 for (int w = 0; w < W; w++) {
//                     int idx_in = ((n*C + c)*H + h)*W + w;
//                     int idx_out = n*flat_size + c*H*W + h*W + w;
//                     out.data[idx_out] = input.data[idx_in];
//                 }
//             }
//         }
//     }

//     return out;
// }

// // [N, C*H*W, 1, 1] -> [N, C, H, W]
// Tensor Flatten::backward(const Tensor& grad_output) {
//     Tensor grad(grad_output.data, input_shape);
//     return grad;
// }

Tensor Flatten::forward(const Tensor& input) {
    input_shape = input.shape; // save original shape [N, C, H, W]
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int flat_size = C * H * W;

    // std::cout << "Flatten forward: N=" << N << ", C=" << C 
    //           << ", H=" << H << ", W=" << W << "\n";
    // std::cout << "Flatten output shape: [" << N << ", " << flat_size << ", 1, 1]\n";

    Tensor out(std::vector<float>(N * flat_size), {N, flat_size, 1, 1});

    // Fill output batch by batch
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    int input_idx = n*C*H*W + c*H*W + i*W + j;
                    int out_idx   = n*flat_size + c*H*W + i*W + j;
                    out.data[out_idx] = input.data[input_idx];
                }
            }
        }
    }

    // std::cout << "flat size: " << flat_size << "\n";

    return out;
}

Tensor Flatten::backward(const Tensor& grad_output) {
    Tensor grad(grad_output.data, grad_output.shape);
    grad.shape = input_shape; // restore original shape [N, C, H, W]
    return grad;
}

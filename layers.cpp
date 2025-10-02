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

std::pair<Tensor, double> Conv2D::forward(const Tensor& input) {
    auto start = std::chrono::high_resolution_clock::now();
    input_cache = input;

    int H = input.shape[1];
    int W = input.shape[2];
    int out_H = (H - kernel_size) / stride + 1;
    int out_W = (W - kernel_size) / stride + 1;

    Tensor out(std::vector<float>(out_channels * out_H * out_W, 0.0f),
               {out_channels, out_H, out_W});

    const float* input_data = input.data.data();
    const float* weights_data = weights.data();
    float* out_data = out.data.data();

    // MUDANÇA: Usando schedule(static) para overhead menor.
    #pragma omp parallel for collapse(3) schedule(static)
    for (int oc = 0; oc < out_channels; oc++) {
        for (int i = 0; i < out_H; i++) {
            for (int j = 0; j < out_W; j++) {
                float sum = bias[oc];
                
                // OTIMIZAÇÃO: Loop sobre os canais de entrada
                for (int ic = 0; ic < in_channels; ic++) {
                    // OTIMIZAÇÃO: Pré-calcula ponteiros para reduzir o cálculo de índice repetido
                    const float* p_input = input_data + (ic * H * W) + (i * stride * W) + (j * stride);
                    const float* p_weights = weights_data + (oc * in_channels + ic) * kernel_size * kernel_size;

                    // O loop mais interno é um produto escalar, ideal para vetorização (SIMD)
                    for (int ki = 0; ki < kernel_size; ki++) {
                        for (int kj = 0; kj < kernel_size; kj++) {
                            sum += p_input[ki * W + kj] * p_weights[ki * kernel_size + kj];
                        }
                    }
                }
                // OTIMIZAÇÃO: Acesso direto à memória de saída
                out_data[(oc * out_H * out_W) + (i * out_W) + j] = sum;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    Conv2D::total_forward_time += duration;
    return {out, duration};
}

std::pair<Tensor, double> Conv2D::backward(const Tensor& grad_output) {
    auto start = std::chrono::high_resolution_clock::now();

    int H = input_cache.shape[1];
    int W = input_cache.shape[2];
    int out_H = grad_output.shape[1];
    int out_W = grad_output.shape[2];

    Tensor grad_input(std::vector<float>(in_channels * H * W, 0.0f), {in_channels, H, W});
    std::fill(grad_weights.data.begin(), grad_weights.data.end(), 0.0f);
    std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);

    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }

    // Thread-private buffers
    std::vector<std::vector<float>> priv_grad_weights(nthreads, std::vector<float>(grad_weights.data.size(), 0.0f));
    std::vector<std::vector<float>> priv_grad_bias(nthreads, std::vector<float>(grad_bias.size(), 0.0f));
    std::vector<std::vector<float>> priv_grad_input(nthreads, std::vector<float>(grad_input.data.size(), 0.0f));

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int oc = 0; oc < out_channels; oc++) {
        for (int i = 0; i < out_H; i++) {
            for (int j = 0; j < out_W; j++) {
                int tid = omp_get_thread_num();
                float go = grad_output.at(oc, i, j);
                priv_grad_bias[tid][oc] += go;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int ki = 0; ki < kernel_size; ki++) {
                        for (int kj = 0; kj < kernel_size; kj++) {
                            int xi = i * stride + ki;
                            int xj = j * stride + kj;
                            int widx = ((oc*in_channels+ic)*kernel_size+ki)*kernel_size + kj;
                            priv_grad_weights[tid][widx] += input_cache.at(ic, xi, xj) * go;
                            int gidx = (ic * H + xi) * W + xj;
                            priv_grad_input[tid][gidx] += weights[widx] * go;
                        }
                    }
                }
            }
        }
    }

    // Reduce thread-private buffers into main gradients
    for (int t = 0; t < nthreads; ++t) {
        for (size_t i = 0; i < grad_weights.data.size(); ++i)
            grad_weights.data[i] += priv_grad_weights[t][i];
        for (size_t i = 0; i < grad_bias.size(); ++i)
            grad_bias[i] += priv_grad_bias[t][i];
        for (size_t i = 0; i < grad_input.data.size(); ++i)
            grad_input.data[i] += priv_grad_input[t][i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    Conv2D::total_backward_time += duration;
    return {grad_input, duration};
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

std::pair<Tensor, double> ReLU::forward(const Tensor& input) {
    auto start = std::chrono::high_resolution_clock::now();

    input_cache = input;
    Tensor out(input.data, input.shape);

    // OTIMIZAÇÃO: Paraleliza o loop sobre todos os elementos do tensor.
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < out.data.size(); idx++) {
        out.data[idx] = std::max(0.0f, out.data[idx]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    ReLU::total_forward_time += duration;
    return {out, duration};
}

std::pair<Tensor, double> ReLU::backward(const Tensor& grad_output) {
    auto start = std::chrono::high_resolution_clock::now();

    Tensor grad(grad_output.data, grad_output.shape);

    // OTIMIZAÇÃO: Paraleliza o loop sobre todos os elementos.
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < grad.data.size(); idx++) {
        grad.data[idx] *= (input_cache.data[idx] > 0 ? 1.0f : 0.0f);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    ReLU::total_backward_time += duration;
    return {grad, duration};
}

// ================================================================== MAXPOOL

double MaxPool2x2::total_forward_time = 0.0;
double MaxPool2x2::total_backward_time = 0.0;

MaxPool2x2::MaxPool2x2() {}

std::string MaxPool2x2::name() const {
    return "MaxPool2x2";
}

std::pair<Tensor, double> MaxPool2x2::forward(const Tensor& input) {
    auto start = std::chrono::high_resolution_clock::now();

    input_cache = input;
    int C = input.shape[0];
    int H = input.shape[1];
    int W = input.shape[2];

    int out_H = H / 2;
    int out_W = W / 2;

    Tensor output(std::vector<float>(C * out_H * out_W, 0.0f), {C, out_H, out_W});
    
    // Vetor pre alocado e mais eficiente que push_back
    max_indices.assign(C * out_H * out_W, 0);

    // OTIMIZAÇÃO: Paraleliza sobre os canais, altura e largura da saída.
    #pragma omp parallel for collapse(3) schedule(static)
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < out_H; i++) {
            for (int j = 0; j < out_W; j++) {
                float max_val = -1e9f;
                int max_idx = -1;

                for (int di = 0; di < 2; di++) {
                    for (int dj = 0; dj < 2; dj++) {
                        int r = i * 2 + di;
                        int col = j * 2 + dj;
                        float val = input.at(c, r, col);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = (c * H + r) * W + col; 
                        }
                    }
                }
                output.at(c, i, j) = max_val;
                max_indices[(c * out_H + i) * out_W + j] = max_idx;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    MaxPool2x2::total_forward_time += duration;
    return {output, duration};
}

std::pair<Tensor, double> MaxPool2x2::backward(const Tensor& grad_output) {
    auto start = std::chrono::high_resolution_clock::now();

    int C = input_cache.shape[0];
    int H = input_cache.shape[1];
    int W = input_cache.shape[2];

    Tensor grad_input(std::vector<float>(C * H * W, 0.0f), {C, H, W});
    int out_H = grad_output.shape[1];
    int out_W = grad_output.shape[2];

    // OTIMIZAÇÃO: O cálculo para cada ponto da saída é independente e escreve
    // em uma posição única da entrada, então a paralelização é segura.
    #pragma omp parallel for collapse(3) schedule(static)
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < out_H; i++) {
            for (int j = 0; j < out_W; j++) {
                int idx = (c * out_H + i) * out_W + j;
                int max_idx = max_indices[idx];
                grad_input.data[max_idx] = grad_output.at(c, i, j);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    MaxPool2x2::total_backward_time += duration;
    return {grad_input, duration};
}

// ================================================================== FULLYCONNECTED
double FullyConnected::total_forward_time = 0.0;
double FullyConnected::total_backward_time = 0.0;


FullyConnected::FullyConnected(int out_size) : out_size(out_size) {}

void FullyConnected::init_from_tensor(const Tensor& input) {
    in_size = 1;
    for (auto dim : input.shape) in_size *= dim;

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

std::pair<Tensor, double> FullyConnected::forward(const Tensor& input) {
    if (in_size == -1) init_from_tensor(input);

    auto start = std::chrono::high_resolution_clock::now();

    input_cache = input;
    Tensor out(std::vector<float>(out_size, 0.0f), {out_size, 1, 1});

    // OTIMIZAÇÃO: O cálculo de cada neurônio de saída é independente.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_size; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_size; j++)
            sum += weights[i * in_size + j] * input.data[j];
        out.data[i] = sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    FullyConnected::total_forward_time += duration;
    return {out, duration};
}

std::pair<Tensor, double> FullyConnected::backward(const Tensor& grad_output) {
    auto start = std::chrono::high_resolution_clock::now();

    Tensor grad_input(std::vector<float>(in_size, 0.0f), {in_size, 1, 1});
    std::fill(grad_weights.begin(), grad_weights.end(), 0.0f);
    std::fill(grad_bias.begin(), grad_bias.end(), 0.0f);

    for (int i = 0; i < out_size; i++) {
        float go = grad_output.data[i];
        grad_bias[i] += go;
        for (int j = 0; j < in_size; j++) {
            grad_weights[i * in_size + j] += input_cache.data[j] * go;
            grad_input.data[j] += weights[i * in_size + j] * go;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    FullyConnected::total_backward_time += duration;
    return {grad_input, duration};
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

std::pair<Tensor, double> Softmax::forward(const Tensor& input) {

    auto start = std::chrono::high_resolution_clock::now();

    output_cache = input; // cache input para debug

    Tensor out(std::vector<float>(input.data.size(), 0.0f),
               {static_cast<int>(input.data.size()), 1, 1});

    float max_val = *std::max_element(input.data.begin(), input.data.end());
    float sum = 0.0f;

    for (size_t i = 0; i < input.data.size(); i++) {
        out.data[i] = std::exp(input.data[i] - max_val);
        sum += out.data[i];
    }

    for (auto& v : out.data) v /= sum;

    output_cache = out; // cache output para debug

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    Softmax::total_forward_time += duration;

    return {out, duration};

}

std::pair<Tensor, double> Softmax::backward(const Tensor& grad_output) {
    // Softmax backward is usually handled with loss, but for now just pass through
    return {grad_output, 0.0};
}

// ================================================================== FLATTEN

// [C,H,W] -> [C*H*W, 1, 1]
std::pair<Tensor, double> Flatten::forward(const Tensor& input) {
    input_shape = input.shape;
    return {Tensor(input.data, { (int)input.data.size(), 1, 1 }), 0.0};
}

// [C*H*W,1,1] -> [C,H,W]
std::pair<Tensor, double> Flatten::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;
    grad.shape = input_shape;
    return {grad, 0.0};
}
#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>

// Base layer
class Layer {
public:
    virtual std::string name() const = 0;
    virtual void debugPrint() const = 0;
    virtual void update(float lr) {}   // default: nothing
    virtual ~Layer() {}

    // nao colocar forward e backward aq pq tem 2d e 1d !
};

// =================== Conv2D ===================
class Conv2D : public Layer {
public:
    int input_dim, kernel_size, num_filters;
    std::vector<std::vector<std::vector<float>>> kernels;     // [num_filters][kH][kW]
    std::vector<std::vector<std::vector<float>>> grad_kernels; // gradientes acumulados
    std::vector<std::vector<float>> last_input; // input armazenado

    static double total_forward_time;   
    static double total_backward_time;
    static double total_update_time;

    Conv2D(int input_dim, int kernel_size, int num_filters);

    std::string name() const override { return "Conv2D"; }

    // forward: input 2D -> output 3D (num_filters mapas)
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<float>>& input);

    // backward: grad_output 3D (mesma forma da saída do forward)
    std::vector<std::vector<float>> backward(const std::vector<std::vector<std::vector<float>>>& grad_output);

    void update(float lr) override;
    void debugPrint() const override;
};

// =================== ReLU ===================
class ReLU : public Layer {
public:
    std::vector<std::vector<float>> last_input;

    static double total_forward_time;   
    static double total_backward_time;

    std::string name() const override { return "ReLU"; }
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& grad_output);
    void debugPrint() const override;
};

// =================== MaxPool2x2 ===================
class MaxPool2x2 : public Layer {
public:
    int input_dim;
    std::vector<std::vector<float>> last_input;
    std::vector<int> max_indices;

    static double total_forward_time;   
    static double total_backward_time;

    MaxPool2x2(int input_dim);
    std::string name() const override { return "MaxPool2x2"; }
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& grad_output);
    void debugPrint() const override;
};

// =================== FullyConnected ===================
class FullyConnected : public Layer {
public:
    int in_size, out_size;
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;

    // gradientes
    std::vector<std::vector<float>> grad_weights;
    std::vector<float> grad_bias;

    std::vector<float> last_input;

    static double total_forward_time;   
    static double total_backward_time;
    static double total_update_time;

    FullyConnected(int in_size, int out_size);

    std::string name() const override { return "FullyConnected"; }
    std::vector<float> forward(const std::vector<float>& input);
    std::vector<float> backward(const std::vector<float>& grad_output);
    void update(float lr) override;
    void debugPrint() const override;
};

// =================== Softmax ===================
class Softmax : public Layer {
public:
    std::vector<float> last_output;

    static double total_forward_time;   
    static double total_backward_time;

    std::string name() const override { return "Softmax"; }
    std::vector<float> forward(const std::vector<float>& input);
    std::vector<float> backward(const std::vector<float>& grad_output);
    void debugPrint() const override;
};

#endif

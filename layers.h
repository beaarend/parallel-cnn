#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>


struct Tensor {
    std::vector<float> data;   // valores flattened
    std::vector<int> shape;    // [C, H, W]

    Tensor() {}
    Tensor(const std::vector<float>& d, const std::vector<int>& s)
        : data(d), shape(s) {}

    // utilitário para acessar em 3D
    inline float& at(int c, int i, int j) {
        return data[(c * shape[1] + i) * shape[2] + j];
    }
    inline float at(int c, int i, int j) const {
        return data[(c * shape[1] + i) * shape[2] + j];
    }
};

class Layer {
public:
    virtual std::string name() const = 0;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update(float lr) {}

    virtual ~Layer() {}
};

class Conv2D : public Layer {
    int in_channels, out_channels, kernel_size, stride;
    std::vector<float> weights;     // [out, in, k, k]
    std::vector<float> bias;        // [out]

    Tensor input_cache;             // para backward
    Tensor grad_weights;            // mesmo shape de weights
    std::vector<float> grad_bias;   // [out]

public:
    Conv2D(int in_c, int out_c, int k, int s=1);

    std::string name() const override;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update(float lr) override;

    static double total_forward_time;
    static double total_backward_time;
};

class ReLU : public Layer {
    Tensor input_cache; // guarda entrada p backward

public:
    ReLU();

    std::string name() const override;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    static double total_forward_time;
    static double total_backward_time;
};

class MaxPool2x2 : public Layer {
    Tensor input_cache;            // salva a entrada para o backward
    std::vector<int> max_indices;  // salva os índices máximos para cada célula

public:
    MaxPool2x2();

    std::string name() const override;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    static double total_forward_time;
    static double total_backward_time;
};

class FullyConnected : public Layer {
    int in_size = -1;
    int out_size;
    std::vector<float> weights;    // flattened [out * in]
    std::vector<float> bias;       // [out]

    Tensor input_cache;            // para backward
    std::vector<float> grad_weights;  // flattened
    std::vector<float> grad_bias;     // [out]

public:
    FullyConnected(int out_size);

    std::string name() const override;

    void init_from_tensor(const Tensor& input);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update(float lr) override;
    
    static double total_forward_time;
    static double total_backward_time;
};

class Softmax : public Layer {
    Tensor output_cache;

public:
    Softmax() {}

    std::string name() const override;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override; 

    static double total_forward_time;
    static double total_backward_time;
};

class Flatten : public Layer {
    std::vector<int> input_shape;

public:
    Flatten() {}

    std::string name() const override { return "Flatten"; }

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};

#endif

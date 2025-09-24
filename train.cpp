#include "train.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ======================== loss ========================
float cross_entropy_loss(const std::vector<float>& probs, int label) {
    float eps = 1e-8f;
    return -std::log(probs[label] + eps);
}

std::vector<float> cross_entropy_grad(const std::vector<float>& probs, int label) {
    std::vector<float> grad = probs;
    grad[label] -= 1.0f; // dL/dz = probs - one_hot
    return grad;
}

std::vector<float> flatten(const std::vector<std::vector<float>>& input2D) {
    std::vector<float> output;
    for (const auto& row : input2D)
        output.insert(output.end(), row.begin(), row.end());
    return output;
}

std::vector<std::vector<float>> flatten_backward(const std::vector<float>& grad1D,
                                                 const std::vector<std::vector<float>>& ref2D) {
    std::vector<std::vector<float>> grad2D(ref2D.size(), std::vector<float>(ref2D[0].size()));
    int idx = 0;
    for (size_t i = 0; i < ref2D.size(); ++i)
        for (size_t j = 0; j < ref2D[0].size(); ++j)
            grad2D[i][j] = grad1D[idx++];
    return grad2D;
}

// flatten volume 3D -> 1D
std::vector<float> flatten3D(const std::vector<std::vector<std::vector<float>>>& input) {
    std::vector<float> flat;
    for (const auto& mat : input) {
        for (const auto& row : mat) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
    }
    return flat;
}

// backward: reconstrói 3D a partir do grad 1D
std::vector<std::vector<std::vector<float>>> flatten3D_backward(
    const std::vector<float>& grad, 
    const std::vector<std::vector<std::vector<float>>>& ref
) {
    std::vector<std::vector<std::vector<float>>> output = ref;
    size_t idx = 0;
    for (auto& mat : output) {
        for (auto& row : mat) {
            for (auto& val : row) {
                val = grad[idx++];
            }
        }
    }
    return output;
}

// ======================== train ========================
void train_epoch(
    Conv2D& conv,
    ReLU& relu,
    MaxPool2x2& pool,
    FullyConnected& fc,
    Softmax& softmax,
    const std::vector<std::vector<std::vector<float>>>& images,
    const std::vector<int>& labels,
    float lr
) {
    float total_loss = 0.0f;
    int correct = 0;

    for (size_t n = 0; n < images.size(); n++) {
        // forward
        auto out1 = conv.forward(images[n]);              // 3D (num_filters × H × W)
        std::vector<std::vector<std::vector<float>>> out2(out1.size());
        std::vector<std::vector<std::vector<float>>> out_pool(out1.size());

        for (size_t f = 0; f < out1.size(); f++) {
            auto relu_out = relu.forward(out1[f]);        // ReLU por filtro (2D)
            out2[f] = relu_out;
            out_pool[f] = pool.forward(relu_out);         // Pool por filtro (2D)
        }

        auto flat = flatten3D(out_pool);                  // flatten 3D -> 1D
        auto out3 = fc.forward(flat);                     // 1D
        auto out4 = softmax.forward(out3);                // 1D

        // loss
        total_loss += cross_entropy_loss(out4, labels[n]);
        int pred = std::distance(out4.begin(), std::max_element(out4.begin(), out4.end()));
        if (pred == labels[n]) correct++;

        // backward
        auto grad = cross_entropy_grad(out4, labels[n]);  // 1D
        grad = softmax.backward(grad);                    // 1D
        grad = fc.backward(grad);                         // 1D
        fc.update(lr);

        auto grad3D = flatten3D_backward(grad, out_pool); // 1D -> 3D (forma de out_pool)

        std::vector<std::vector<std::vector<float>>> grad_relu(out1.size());
        for (size_t f = 0; f < out1.size(); f++) {
            auto grad_pool = pool.backward(grad3D[f]);    // unpool grad
            grad_relu[f] = relu.backward(grad_pool);      // ReLU backward
        }

        auto grad_input = conv.backward(grad_relu);       // 2D grad
        conv.update(lr);
    }

    std::cout << "Train Loss: " << total_loss / images.size()
              << " | Acc: " << (float)correct / images.size() << "\n";
}


// ======================== eval ========================
float evaluate(
    Conv2D& conv,
    ReLU& relu,
    MaxPool2x2& pool,
    FullyConnected& fc,
    Softmax& softmax,
    const std::vector<std::vector<std::vector<float>>>& images,
    const std::vector<int>& labels
){
    int correct = 0;
    for (size_t n = 0; n < images.size(); n++) {
        auto out1 = conv.forward(images[n]); // 3D

        std::vector<std::vector<std::vector<float>>> out2(out1.size());
        std::vector<std::vector<std::vector<float>>> out_pool(out1.size());

        for (size_t f = 0; f < out1.size(); f++) {
            auto relu_out = relu.forward(out1[f]);
            out2[f] = relu_out;
            out_pool[f] = pool.forward(relu_out);         // aplica pooling
        }

        auto flat = flatten3D(out_pool);     // 3D -> 1D
        auto out3 = fc.forward(flat);        // 1D
        auto out4 = softmax.forward(out3);   // 1D

        int pred = std::distance(out4.begin(), std::max_element(out4.begin(), out4.end()));
        if (pred == labels[n]) correct++;
    }
    return (float)correct / images.size();
}

int predict(
    Conv2D& conv,
    ReLU& relu,
    MaxPool2x2& pool,
    FullyConnected& fc,
    Softmax& softmax,
    const std::vector<std::vector<float>>& image // <-- 2D
) {

    auto conv_out = conv.forward(image); 
    std::vector<std::vector<std::vector<float>>> relu_out_volume(conv_out.size());
    std::vector<std::vector<std::vector<float>>> pool_out_volume(conv_out.size());

    for (size_t f = 0; f < conv_out.size(); f++) {
        auto relu_out = relu.forward(conv_out[f]);
        relu_out_volume[f] = relu_out;

        auto pool_out = pool.forward(relu_out);
        pool_out_volume[f] = pool_out;
    }

    auto flat_vector = flatten3D(pool_out_volume);
    auto fc_out = fc.forward(flat_vector);
    auto probabilities = softmax.forward(fc_out);

    int prediction = std::distance(probabilities.begin(),
                                   std::max_element(probabilities.begin(), probabilities.end()));

    return prediction;
}
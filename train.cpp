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
        auto out1 = conv.forward(images[n]);              // 3D
        std::vector<std::vector<std::vector<float>>> out2(out1.size());
        for (size_t f = 0; f < out1.size(); f++)
            out2[f] = relu.forward(out1[f]);             // ReLU por filtro (2D)

        auto flat = flatten3D(out2);                     // flatten 3D -> 1D
        auto out3 = fc.forward(flat);                    // 1D
        auto out4 = softmax.forward(out3);               // 1D

        // loss
        total_loss += cross_entropy_loss(out4, labels[n]);
        int pred = std::distance(out4.begin(), std::max_element(out4.begin(), out4.end()));
        if (pred == labels[n]) correct++;

        // backward
        auto grad = cross_entropy_grad(out4, labels[n]); // 1D
        grad = softmax.backward(grad);                   // 1D
        grad = fc.backward(grad);                        // 1D
        fc.update(lr);

        auto grad3D = flatten3D_backward(grad, out2);    // 1D -> 3D
        std::vector<std::vector<std::vector<float>>> grad_relu(out2.size());
        for (size_t f = 0; f < out2.size(); f++)
            grad_relu[f] = relu.backward(grad3D[f]);     // 2D por filtro

        auto grad_input = conv.backward(grad_relu);      // 2D
        conv.update(lr);
    }

    std::cout << "Train Loss: " << total_loss / images.size()
              << " | Acc: " << (float)correct / images.size() << "\n";
}


// ======================== eval ========================
float evaluate(
    Conv2D& conv,
    ReLU& relu,
    FullyConnected& fc,
    Softmax& softmax,
    const std::vector<std::vector<std::vector<float>>>& images,
    const std::vector<int>& labels
){
    int correct = 0;
    for (size_t n = 0; n < images.size(); n++) {
        auto out1 = conv.forward(images[n]);        // 2D
        auto out2 = relu.forward(out1);            // 2D
        auto flat = flatten(out2);                 // 1D
        auto out3 = fc.forward(flat);              // 1D
        auto out4 = softmax.forward(out3);         // 1D

        int pred = std::distance(out4.begin(), std::max_element(out4.begin(), out4.end()));
        if (pred == labels[n]) correct++;
    }
    return (float)correct / images.size();
}

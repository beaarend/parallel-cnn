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

void train_epoch(
    std::vector<Layer*>& layers,
    const std::vector<std::vector<std::vector<float>>>& images,
    const std::vector<int>& labels,
    float lr
) {
    float total_loss = 0.0f;
    int correct = 0;
    double total_forward_time = 0.0;
    double total_backward_time = 0.0;

    // std::cout << "entrou train epoch\n";

    for (size_t n = 0; n < images.size(); n++) {
        // wrap input as Tensor (1 channel, H, W)
        int h = images[n].size();
        int w = images[n][0].size();

        // std::cout << "imagem " << n << " tamanho " << h << "x" << w << "\n";

        Tensor x(flatten3D({images[n]}), {1, h, w});

        // std::cout << "depois flatten\n";

        std::vector<Tensor> activations;
        activations.push_back(x);
        double sample_forward_time = 0.0;

        // forward pass
        for (auto* layer : layers) {
            auto [out, dur] = layer->forward(x);
            x = out;
            activations.push_back(x);
            sample_forward_time += dur;
        }
        total_forward_time += sample_forward_time;

        // loss
        auto& probs = activations.back().data;
        total_loss += cross_entropy_loss(probs, labels[n]);
        int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        if (pred == labels[n]) correct++;

        // backward pass
        Tensor grad(cross_entropy_grad(probs, labels[n]), {(int)probs.size()});
        double sample_backward_time = 0.0;
        for (int i = layers.size() - 1; i >= 0; i--) {
            auto [g, dur] = layers[i]->backward(grad);
            grad = g;
            layers[i]->update(lr);
            sample_backward_time += dur;
        }
        total_backward_time += sample_backward_time;
    }

    std::cout << "Train Loss: " << total_loss / images.size()
              << " | Acc: " << (float)correct / images.size() << "\n";
    std::cout << "Total Forward Time (ms): " << total_forward_time << "\n";
    std::cout << "Total Backward Time (ms): " << total_backward_time << "\n";
}

// ======================== evaluate ========================
void evaluate(const std::vector<Layer*>& layers,
               const std::vector<std::vector<std::vector<float>>>& images,
               const std::vector<int>& labels) 
{
    int correct = 0;

    for (size_t n = 0; n < images.size(); n++) {
        int h = images[n].size();
        int w = images[n][0].size();

        Tensor x(flatten3D({images[n]}), {1, h, w});

        // forward pass through all layers
        for (auto* layer : layers) {
            auto [out, not_used] = layer->forward(x);
            x = out;
        }

        auto& probs = x.data;
        int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        if (pred == labels[n]) correct++;
    }

    std::cout << "Eval Acc: " << (float)correct / images.size() << "\n";
}

// ======================== predict ========================
int predict(const std::vector<Layer*>& layers,
            const std::vector<std::vector<float>>& image2D) 
{
    int h = image2D.size();
    int w = image2D[0].size();

    Tensor x(flatten({image2D}), {1, h, w});

    // forward pass through all layers
    for (auto* layer : layers) {
        auto [out, not_used] = layer->forward(x);
        x = out;
    }

    auto& probs = x.data;
    int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    return pred;
}
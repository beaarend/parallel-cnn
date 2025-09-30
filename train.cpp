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

        // std::cout << "depois de activation\n";

        // forward pass
        for (auto* layer : layers) {
            // std::cout << "Layer: " << layer->name() << "\n";
            x = layer->forward(x);
            activations.push_back(x);
        }

        // loss
        auto& probs = activations.back().data;
        total_loss += cross_entropy_loss(probs, labels[n]);
        int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        if (pred == labels[n]) correct++;

        // backward pass
        Tensor grad(cross_entropy_grad(probs, labels[n]), {(int)probs.size()});
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers[i]->backward(grad);
            layers[i]->update(lr);
        }
    }

    std::cout << "Train Loss: " << total_loss / images.size()
              << " | Acc: " << (float)correct / images.size() << "\n";
}

// void train_epoch_batch(
//     std::vector<Layer*>& layers,
//     const std::vector<std::vector<std::vector<float>>>& images,
//     const std::vector<int>& labels,
//     float lr,
//     int batch_size
// ) {
//     float total_loss = 0.0f;
//     int correct = 0;

//     for (size_t start = 0; start < images.size(); start += batch_size) {
//         size_t end = std::min(start + batch_size, images.size());
//         size_t current_batch = end - start;

//         // ==== Prepare batch tensors ====
//         std::vector<Tensor> batch_inputs;
//         batch_inputs.reserve(current_batch);
//         std::vector<int> batch_labels;

//         for (size_t n = start; n < end; n++) {
//             int h = images[n].size();
//             int w = images[n][0].size();
//             Tensor x(flatten3D({images[n]}), {1, h, w}); // 1 channel
//             batch_inputs.push_back(x);
//             batch_labels.push_back(labels[n]);
//         }

//         // ==== Forward pass ====
//         std::vector<Tensor> batch_activations = batch_inputs;

//         for (auto* layer : layers) {
//             // Parallelize inside layers like Conv2D / MaxPool2x2
//             for (auto& t : batch_activations) {
//                 t = layer->forward(t);
//             }
//         }

//         // ==== Compute loss & backward per sample ====
//         for (size_t i = 0; i < batch_activations.size(); i++) {
//             auto& probs = batch_activations[i].data;
//             total_loss += cross_entropy_loss(probs, batch_labels[i]);

//             int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
//             if (pred == batch_labels[i]) correct++;

//             // Backward pass
//             Tensor grad(cross_entropy_grad(probs, batch_labels[i]), {(int)probs.size()});
//             for (int j = layers.size() - 1; j >= 0; j--) {
//                 grad = layers[j]->backward(grad);
//                 layers[j]->update(lr);
//             }
//         }
//     }

//     std::cout << "Train Loss: " << total_loss / images.size()
//               << " | Acc: " << (float)correct / images.size() << "\n";
// }

void train_epoch_batch(
    std::vector<Layer*>& layers,
    const std::vector<std::vector<std::vector<float>>>& images,
    const std::vector<int>& labels,
    float lr,
    int batch_size
) {
    float total_loss = 0.0f;
    int correct = 0;

    int N_total = images.size();
    int C = 1;                      // grayscale
    int H = images[0].size();
    int W = images[0][0].size();

    std::cout << "Total images: " << N_total << ", Batch size: " << batch_size << "\n";

    for (size_t start = 0; start < images.size(); start += batch_size) {
        size_t end = std::min(start + batch_size, images.size());
        size_t current_batch = end - start;

        // ==== Prepare batch tensor [N,C,H,W] ====
        Tensor batch_input(std::vector<float>(current_batch * C * H * W, 0.0f),
                           { (int)current_batch, C, H, W });

        std::cout << "Processing batch from " << start << " to " << end - 1
                  << " (size " << current_batch << ")\n";

        for (size_t n = 0; n < current_batch; n++) {
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    batch_input.data[n*C*H*W + i*W + j] = images[start + n][i][j];
                }
            }
        }

        // ==== Forward pass (batch-aware) ====
        Tensor batch_activations = batch_input;
        for (auto* layer : layers) {
            // std::cout << "Layer: " << layer->name() << "\n";
            batch_activations = layer->forward(batch_activations); // layers must handle 4D
        }

        // std::cout << "sobrevivi a forward do batch\n";

        // ==== Compute loss & backward per sample ====
        for (size_t n = 0; n < current_batch; n++) {
            // Extract single sample from batch tensor
            std::vector<float> probs;
            int out_size = batch_activations.shape[1] * batch_activations.shape[2] * batch_activations.shape[3];
            probs.reserve(out_size);

            int offset = n * out_size;
            for (int i = 0; i < out_size; i++) probs.push_back(batch_activations.data[offset + i]);

            total_loss += cross_entropy_loss(probs, labels[start + n]);

            int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
            if (pred == labels[start + n]) correct++;

            // Backward
            Tensor grad(cross_entropy_grad(probs, labels[start + n]), {1, (int)probs.size(), 1, 1});
            for (int j = layers.size() - 1; j >= 0; j--) {
                grad = layers[j]->backward(grad);
                layers[j]->update(lr); // still accumulates gradients
            }
        }
    }

    std::cout << "Train Loss: " << total_loss / N_total
              << " | Acc: " << (float)correct / N_total << "\n";
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
            x = layer->forward(x);
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
        x = layer->forward(x);
    }

    auto& probs = x.data;
    int pred = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    return pred;
}
#include "train.h"
#include <omp.h>
#include <iostream>
#include <algorithm>

// helper to flatten 3D -> 1D (already in your file — reuse)
std::vector<float> flatten3D(const std::vector<std::vector<std::vector<float>>>& input);
std::vector<std::vector<std::vector<float>>> flatten3D_backward(
    const std::vector<float>& grad,
    const std::vector<std::vector<std::vector<float>>>& ref
);

// thread-parallel train_epoch
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
    int N = images.size();

    int num_threads = omp_get_max_threads();

    // allocate per-thread accumulators
    int num_filters = conv.num_filters;
    int ksz = conv.kernel_size;
    int fc_out = fc.out_size;
    int fc_in = fc.in_size;

    // per-thread conv grads: [num_threads][num_filters][kH][kW]
    std::vector<std::vector<std::vector<std::vector<float>>>> conv_thread_grads(
        num_threads,
        std::vector<std::vector<std::vector<float>>>(num_filters,
            std::vector<std::vector<float>>(ksz, std::vector<float>(ksz, 0.0f)))
    );

    // per-thread fc grads
    std::vector<std::vector<std::vector<float>>> fc_thread_grad_wt(
        num_threads, std::vector<std::vector<float>>(fc_out, std::vector<float>(fc_in, 0.0f))
    );
    std::vector<std::vector<float>> fc_thread_grad_bias(num_threads, std::vector<float>(fc_out, 0.0f));

    // reductions for loss and accuracy will use OpenMP reduction
    #pragma omp parallel for schedule(dynamic) reduction(+:total_loss, correct)
    for (int n = 0; n < N; ++n) {
        int tid = omp_get_thread_num();

        // ---- FORWARD (thread-local storages) ----
        std::vector<std::vector<float>> conv_last_input;
        auto conv_out = conv.forward_nostate(images[n], conv_last_input); // 3D [num_filters][H][W]

        // per-filter relu/pool storages
        std::vector<std::vector<std::vector<float>>> relu_out_volume(conv_out.size());
        std::vector<std::vector<std::vector<float>>> pool_out_volume(conv_out.size());
        std::vector<std::vector<int>> pool_max_indices(conv_out.size());

        for (size_t f = 0; f < conv_out.size(); f++) {
            std::vector<std::vector<float>> relu_last;
            auto relu_out = relu.forward_nostate_relu(conv_out[f], relu_last);
            relu_out_volume[f] = relu_out;

            std::vector<int> maxidx;
            auto pool_out = pool.forward_nostate(relu_out, maxidx);
            pool_out_volume[f] = pool_out;
            pool_max_indices[f] = std::move(maxidx);
        }

        auto flat = flatten3D(pool_out_volume);
        std::vector<float> fc_last_input;
        auto out3 = fc.forward_nostate(flat, fc_last_input);
        std::vector<float> soft_last;
        auto out4 = softmax.forward_nostate(out3, soft_last);

        // loss & accuracy
        total_loss += cross_entropy_loss(out4, labels[n]);
        int pred = std::distance(out4.begin(), std::max_element(out4.begin(), out4.end()));
        if (pred == labels[n]) ++correct;

        // ---- BACKWARD (compute gradients into thread-local accumulators) ----
        // cross-entropy grad
        auto grad = cross_entropy_grad(out4, labels[n]);
        grad = softmax.backward_nostate(grad, soft_last); // identity mostly
        // FC backward -> produces grad for FC weights & grad_input (1D) and does NOT modify fc internals
        std::vector<std::vector<float>> fc_local_grad_w;
        std::vector<float> fc_local_grad_b;
        auto grad_from_fc = fc.backward_nostate(grad, fc_last_input, fc_local_grad_w, fc_local_grad_b);

        // accumulate into per-thread FC grads
        for (int i = 0; i < fc_out; ++i) {
            fc_thread_grad_bias[tid][i] += fc_local_grad_b[i];
            for (int j = 0; j < fc_in; ++j) {
                fc_thread_grad_wt[tid][i][j] += fc_local_grad_w[i][j];
            }
        }

        // reconstruct 3D grad from flattened grad_from_fc
        auto grad3D = flatten3D_backward(grad_from_fc, pool_out_volume);

        // for each filter: pool backward (needs max_indices stored), relu backward, conv backward to get input grad and per-filter kernel grads
        // We'll accumulate per-filter kernel grads into conv_thread_grads[tid]
        std::vector<std::vector<std::vector<float>>> grad_relu(conv_out.size());
        for (size_t f = 0; f < conv_out.size(); f++) {
            // pool backward
            auto grad_pool = pool.backward_nostate(grad3D[f], pool_max_indices[f]);
            // relu backward (requires last_input which is relu input = conv_out[f])
            auto grad_r = relu.backward_nostate(grad_pool, conv_out[f]); // here conv_out[f] acted as relu last_input
            grad_relu[f] = grad_r;
        }

        // conv backward: compute grad_input (ignored) and produce per-filter kernel gradients
        std::vector<std::vector<std::vector<float>>> conv_local_grad_kernels;
        auto grad_input = conv.backward_nostate(grad_relu, conv_last_input, conv_local_grad_kernels);

        // accumulate conv_local_grad_kernels into thread accumulator
        for (int f = 0; f < conv.num_filters; ++f) {
            for (int ki = 0; ki < conv.kernel_size; ++ki) {
                for (int kj = 0; kj < conv.kernel_size; ++kj) {
                    conv_thread_grads[tid][f][ki][kj] += conv_local_grad_kernels[f][ki][kj];
                }
            }
        }
    } // end parallel for images

    // ---- Reduce per-thread grads into model's grads ----
    // Zero model grads first
    for (int f = 0; f < conv.num_filters; ++f)
        for (int i = 0; i < conv.kernel_size; ++i)
            for (int j = 0; j < conv.kernel_size; ++j)
                conv.grad_kernels[f][i][j] = 0.0f;

    for (int tid = 0; tid < num_threads; ++tid) {
        for (int f = 0; f < conv.num_filters; ++f) {
            for (int ki = 0; ki < conv.kernel_size; ++ki)
                for (int kj = 0; kj < conv.kernel_size; ++kj)
                    conv.grad_kernels[f][ki][kj] += conv_thread_grads[tid][f][ki][kj];
        }
    }

    // FC grads reduce
    for (int i = 0; i < fc.out_size; ++i) {
        for (int j = 0; j < fc.in_size; ++j) fc.grad_weights[i][j] = 0.0f;
        fc.grad_bias[i] = 0.0f;
    }
    for (int tid = 0; tid < num_threads; ++tid) {
        for (int i = 0; i < fc.out_size; ++i) {
            fc.grad_bias[i] += fc_thread_grad_bias[tid][i];
            for (int j = 0; j < fc.in_size; ++j)
                fc.grad_weights[i][j] += fc_thread_grad_wt[tid][i][j];
        }
    }

    float scale = 1.0f / N; // N = number of images
for (int f = 0; f < conv.num_filters; ++f)
    for (int i = 0; i < conv.kernel_size; ++i)
        for (int j = 0; j < conv.kernel_size; ++j)
            conv.grad_kernels[f][i][j] *= scale;

for (int i = 0; i < fc.out_size; ++i) {
    fc.grad_bias[i] *= scale;
    for (int j = 0; j < fc.in_size; ++j)
        fc.grad_weights[i][j] *= scale;
}


    // finally update model (single-thread)
    conv.update(lr);
    fc.update(lr);

    std::cout << "Train Loss: " << total_loss / N
              << " | Acc: " << (float)correct / N << "\n";
}


float evaluate(
    Conv2D& conv,
    ReLU& relu,
    MaxPool2x2& pool,
    FullyConnected& fc,
    Softmax& softmax,
    const std::vector<std::vector<std::vector<float>>>& images,
    const std::vector<int>& labels
) {
    int correct = 0;
    for (size_t n = 0; n < images.size(); n++) {
        auto out1 = conv.forward(images[n]);

        std::vector<std::vector<std::vector<float>>> out2(out1.size());
        std::vector<std::vector<std::vector<float>>> out_pool(out1.size());

        for (size_t f = 0; f < out1.size(); f++) {
            auto relu_out = relu.forward(out1[f]);
            out2[f] = relu_out;
            out_pool[f] = pool.forward(relu_out);
        }

        auto flat = flatten3D(out_pool);
        auto out3 = fc.forward(flat);
        auto out4 = softmax.forward(out3);

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
    const std::vector<std::vector<float>>& image
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

float cross_entropy_loss(const std::vector<float>& probs, int label) {
    float eps = 1e-8f;
    return -std::log(probs[label] + eps);
}

std::vector<float> cross_entropy_grad(const std::vector<float>& probs, int label) {
    std::vector<float> grad = probs;
    grad[label] -= 1.0f; // dL/dz = probs - one_hot
    return grad;
}

std::vector<float> flatten3D(const std::vector<std::vector<std::vector<float>>>& input) {
    std::vector<float> flat;
    for (const auto& mat : input) {
        for (const auto& row : mat) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
    }
    return flat;
}

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


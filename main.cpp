#include "layers.h"
#include "train.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <omp.h>
#include <algorithm>  // para std::shuffle, std::iota, etc.
#include <numeric>    // para std::iota

void load_dataset_csv(const std::string& file_path,
                    std::vector<std::vector<std::vector<float>>>& images,
                    std::vector<int>& labels,
                    int num_samples = -1) 
{
    std::ifstream file(file_path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + file_path);

    images.clear();
    labels.clear();

    std::cout << "Loading dataset data from " << file_path << "...\n";

    std::string line;
    int count = 0;

    if (!std::getline(file, line))
        throw std::runtime_error("CSV is empty: " + file_path);

    while (std::getline(file, line)) {
        if (num_samples > 0 && count >= num_samples) break;

        // std::cout << "\rLoaded " << count + 1 << " samples \n" << std::flush;

        std::stringstream ss(line);
        std::string value;

        // First value is the label
        if (!std::getline(ss, value, ','))
            throw std::runtime_error("CSV parsing error on line " + std::to_string(count+1));
        labels.push_back(std::stoi(value));

        // Remaining 784 values are pixels
        std::vector<std::vector<float>> image(28, std::vector<float>(28));
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                if (!std::getline(ss, value, ','))
                    throw std::runtime_error("CSV parsing error on line " + std::to_string(count+1));
                image[r][c] = std::stof(value) / 255.0f; // normalize to 0..1
            }
        }

        images.push_back(image);
        count++;
    }
}

void load_cifar_csv(const std::string& file_path,
                    std::vector<std::vector<std::vector<std::vector<float>>>>& images,
                    std::vector<int>& labels,
                    int num_samples = -1)
{
    std::ifstream file(file_path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + file_path);

    images.clear();
    labels.clear();

    std::string line;
    int count = 0;

    // --- Pular cabeçalho ---
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (num_samples > 0 && count >= num_samples) break;

        std::stringstream ss(line);
        std::string value;
        std::vector<float> pixels;

        // lê todos os valores da linha
        while (std::getline(ss, value, ',')) {
            pixels.push_back(std::stof(value));
        }

        // if (pixels.size() != 3073)
        //     throw std::runtime_error("CSV parsing error on line " + std::to_string(count+1) + 
        //                              ": expected 3073 values, got " + std::to_string(pixels.size()));
        // último valor é o label
        int lbl = static_cast<int>(pixels.back());
        labels.push_back(lbl);
        pixels.pop_back(); // remove label

        // reshape em [C,H,W]
        std::vector<std::vector<std::vector<float>>> img(3,
            std::vector<std::vector<float>>(32, std::vector<float>(32)));

        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    img[c][i][j] = pixels[idx++] / 255.0f; // normalizado
                }
            }
        }

        images.push_back(img);
        count++;
    }
}

void print_timing(){
    std::cout << "=== Timing ===\n";
    std::cout << "Conv2D forward total: " << Conv2D::total_forward_time << " ms\n";
    std::cout << "Conv2D backward total: " << Conv2D::total_backward_time << " ms\n";
    std::cout << "ReLU forward total: " << ReLU::total_forward_time << " ms\n";
    std::cout << "ReLU backward total: " << ReLU::total_backward_time << " ms\n";
    std::cout << "MaxPool2x2 forward total: " << MaxPool2x2::total_forward_time << " ms\n";
    std::cout << "MaxPool2x2 backward total: " << MaxPool2x2::total_backward_time << " ms\n";
    std::cout << "FullyConnected forward total: " << FullyConnected::total_forward_time << " ms\n";
    std::cout << "FullyConnected backward total: " << FullyConnected::total_backward_time << " ms\n";
    std::cout << "Softmax forward total: " << Softmax::total_forward_time << " ms\n";
    std::cout << "Softmax backward total: " << Softmax::total_backward_time << " ms\n";
}

void count_labels(const std::vector<int>& labels) {
    std::vector<int> counts(10, 0);
    for (int lbl : labels) {
        if (lbl >= 0 && lbl < 10)
            counts[lbl]++;
    }
    std::cout << "Label distribution:\n";
    for (int i = 0; i < counts.size(); i++) {
        std::cout << "Label " << i << ": " << counts[i] << "\n";
    }
}

void split_dataset(
    std::vector<std::vector<std::vector<std::vector<float>>>>& train_images,
    std::vector<int>& train_labels,
    std::vector<std::vector<std::vector<std::vector<float>>>>& test_images,
    std::vector<int>& test_labels,
    float test_ratio // ex: 0.2 = 20% para teste
) {
    if (train_images.size() != train_labels.size())
        throw std::runtime_error("train_images e train_labels têm tamanhos diferentes!");

    int total = train_images.size();
    int n_test = static_cast<int>(total * test_ratio);

    // índices embaralhados
    std::vector<int> indices(total);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // pega os primeiros n_test para teste
    for (int i = 0; i < n_test; i++) {
        int idx = indices[i];
        test_images.push_back(train_images[idx]);
        test_labels.push_back(train_labels[idx]);
    }

    // cria novos vetores de treino apenas com os restantes
    std::vector<std::vector<std::vector<std::vector<float>>>> new_train_images;
    std::vector<int> new_train_labels;

    for (int i = n_test; i < total; i++) {
        int idx = indices[i];
        new_train_images.push_back(train_images[idx]);
        new_train_labels.push_back(train_labels[idx]);
    }

    // substitui os originais
    train_images = std::move(new_train_images);
    train_labels = std::move(new_train_labels);
}

int main() {
    int epochs = 10;
    int n_images = 20;
    srand(42); // 
    // omp_set_num_threads(2);

    std::vector<Layer*> layers;
    // layers.push_back(new Conv2D(3, 64, 3, 1));
    // layers.push_back(new ReLU());
    // layers.push_back(new MaxPool2x2());
    // layers.push_back(new Flatten());
    // layers.push_back(new FullyConnected(64));
    // layers.push_back(new ReLU());
    // layers.push_back(new FullyConnected(10));
    // layers.push_back(new Softmax());
    layers.push_back(new Conv2D(3, 16, 3, 1));
    layers.push_back(new ReLU());
    layers.push_back(new MaxPool2x2());      // 32 -> 16
    layers.push_back(new Conv2D(16, 32, 3, 1));
    layers.push_back(new ReLU());
    layers.push_back(new MaxPool2x2());      // 16 -> 8
    layers.push_back(new Flatten());
    layers.push_back(new FullyConnected(64));
    layers.push_back(new ReLU());
    layers.push_back(new FullyConnected(10));
    layers.push_back(new Softmax());

    std::vector<std::vector<std::vector<std::vector<float>>>> train_images;
    std::vector<int> train_labels;
    load_cifar_csv("dataset/cifar10_train.csv", train_images, train_labels, 10000);

    std::vector<std::vector<std::vector<std::vector<float>>>> test_images;
    std::vector<int> test_labels;
    split_dataset(train_images, train_labels, test_images, test_labels, 0.2f);

    for (int e = 0; e < epochs; e++){
        std::cout << "Epoch " << e+1 << "\n";
        train_epoch(layers, train_images, train_labels, 0.01f);
        evaluate(layers, test_images, test_labels);
    }

    print_timing();

    for (int i=0; i < n_images; i++){
        int idx = rand() % test_images.size();
        int true_lbl = test_labels[idx];
        int pred_lbl = predict(layers, test_images[idx]);
        std::cout << "Random Test Image [" << idx << "]: True Label = " << true_lbl
                  << ", Predicted = " << pred_lbl
                  << (pred_lbl == true_lbl ? " ✅" : " ❌") << "\n";
    }

    return 0;
}

#include "layers.h"
#include "train.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <omp.h>

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

int main() {
    int epochs = 10;
    int n_images = 20;
    srand(42); // 
    omp_set_num_threads(1);

    // std::vector<Layer*> layers;
    // layers.push_back(new Conv2D(1, 3, 8, 1));  // in_channels=1, out_channels=8, kernel=3, stride=1
    // layers.push_back(new ReLU());
    // layers.push_back(new MaxPool2x2());
    // layers.push_back(new Flatten());
    // layers.push_back(new FullyConnected(10)); // n_classes=10
    // layers.push_back(new Softmax());

    // std::vector<Layer*> layers;
    // layers.push_back(new Conv2D(1, 32, 5, 1)); 
    // layers.push_back(new ReLU());
    // layers.push_back(new MaxPool2x2());
    // layers.push_back(new Flatten());
    // layers.push_back(new FullyConnected(64));
    // layers.push_back(new ReLU());
    // layers.push_back(new FullyConnected(10));
    // layers.push_back(new Softmax());
/*
    std::vector<Layer*> layers;
    layers.push_back(new Conv2D(1, 8, 3, 1));  // 1→8 channels
    layers.push_back(new ReLU());
    layers.push_back(new MaxPool2x2());
    layers.push_back(new Conv2D(8, 16, 3, 1)); // 8→16 channels
    layers.push_back(new ReLU());
    layers.push_back(new MaxPool2x2());
    layers.push_back(new Flatten());
    layers.push_back(new FullyConnected(64)); // hidden layer
    layers.push_back(new ReLU());
    layers.push_back(new FullyConnected(10));
    layers.push_back(new Softmax());
*/
    std::vector<Layer*> layers;
    layers.push_back(new Conv2D(1, 128, 5, 1)); 
    layers.push_back(new ReLU());
    layers.push_back(new MaxPool2x2());
    layers.push_back(new Flatten());
    layers.push_back(new FullyConnected(64));
    layers.push_back(new ReLU());
    layers.push_back(new FullyConnected(10));
    layers.push_back(new Softmax());

    std::vector<std::vector<std::vector<float>>> train_images;
    std::vector<int> train_labels;
    load_dataset_csv("dataset/mnist_train.csv", train_images, train_labels, 10000);

    std::vector<std::vector<std::vector<float>>> test_images;
    std::vector<int> test_labels;
    load_dataset_csv("dataset/mnist_test.csv", test_images, test_labels, 2000);

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

    exit(0);

    

    // // Treino
    // int epochs = 100;
    // float lr = 0.01f;

    // for (int e = 0; e < epochs; e++) {
    //     train_epoch(conv1, relu1, pool1, fc1, softmax, train_images, train_labels, lr); 
    //     float acc = evaluate(conv1, relu1, pool1, fc1, softmax, test_images, test_labels); 
    // }

    // for (int i = 0; i < 10; i++) {
    //     int idx = rand() % test_images.size();
    //     int true_lbl = test_labels[idx];
    //     int pred_lbl = predict(conv1, relu1, pool1, fc1, softmax, test_images[idx]);
    //     std::cout << "Random Test Image [" << idx << "]: True Label = " << true_lbl
    //               << ", Predicted = " << pred_lbl
    //               << (pred_lbl == true_lbl ? " ✅" : " ❌") << "\n";
    // }

    // std::cout << "=== Timing ===\n";
    // std::cout << "Conv2D forward total: " << Conv2D::total_forward_time << " ms\n";
    // std::cout << "Conv2D backward total: " << Conv2D::total_backward_time << " ms\n";
    // std::cout << "ReLU forward total: " << ReLU::total_forward_time << " ms\n";
    // std::cout << "ReLU backward total: " << ReLU::total_backward_time << " ms\n";
    // std::cout << "MaxPool2x2 forward total: " << MaxPool2x2::total_forward_time << " ms\n";
    // std::cout << "MaxPool2x2 backward total: " << MaxPool2x2::total_backward_time << " ms\n";
    // std::cout << "FullyConnected forward total: " << FullyConnected::total_forward_time << " ms\n";
    // std::cout << "FullyConnected backward total: " << FullyConnected::total_backward_time << " ms\n";
    // std::cout << "Softmax forward total: " << Softmax::total_forward_time << " ms\n";
    // std::cout << "Softmax backward total: " << Softmax::total_backward_time << " ms\n";

    return 0;
}

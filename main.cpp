#include "layers.h"
#include "train.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

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

int main() {
    Conv2D conv1(28, 3, 1);   // entrada 28x28, kernel 3x3, 1 filtro
    MaxPool2x2 pool1(26); // entrada 26x26 (após conv 3x3)
    FullyConnected fc1(13 * 13 * 1, 10); // saida 10 classes
    ReLU relu1;
    Softmax softmax;

    std::vector<std::vector<std::vector<float>>> train_images;
    std::vector<int> train_labels;
    load_dataset_csv("dataset/mnist_train.csv", train_images, train_labels, 10000);

    std::vector<std::vector<std::vector<float>>> test_images;
    std::vector<int> test_labels;
    load_dataset_csv("dataset/mnist_test.csv", test_images, test_labels, 2000);

    // Treino
    int epochs = 100;
    float lr = 0.01f;

    for (int e = 0; e < epochs; e++) {
        // std::cout << "Epoch " << e+1 << "\n";
        train_epoch(conv1, relu1, pool1, fc1, softmax, train_images, train_labels, lr); 
        float acc = evaluate(conv1, relu1, pool1, fc1, softmax, test_images, test_labels); 
        // std::cout << "Test Acc: " << acc << "\n";  
    }

    for (int i = 0; i < 10; i++) {
        int idx = rand() % test_images.size();
        int true_lbl = test_labels[idx];
        int pred_lbl = predict(conv1, relu1, pool1, fc1, softmax, test_images[idx]);
        std::cout << "Random Test Image [" << idx << "]: True Label = " << true_lbl
                  << ", Predicted = " << pred_lbl
                  << (pred_lbl == true_lbl ? " ✅" : " ❌") << "\n";
    }

    return 0;
}

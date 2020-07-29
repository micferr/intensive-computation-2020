#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <thread>
#include <vector>

template<typename T>
void print_vector(const std::vector<T> v) {
    for (const auto& element : v) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

template<typename T>
int reduction_sum(const std::vector<T>& v, int rows, int cols) {
    std::vector<int> partial_sums(rows, 0);
    std::vector<std::thread> threads;

    auto partial_sum = [&v, cols, &partial_sums](int row) {
        int acc = 0;
        for (int j = 0; j < cols; j++) {
            acc += v[row*cols + j];
        }
        partial_sums[row] = acc;
    };

    for (int i = 0; i < rows; i++) {
        std::thread t(partial_sum, i);
        threads.emplace_back(std::move(t));
    }
    for (auto& t : threads) t.join();

    auto acc = 0;
    for (const auto part_sum : partial_sums) {
        acc += part_sum;
    }
    return acc;
}

template<typename T>
void fast_segmented_sum(std::vector<T>& v, std::vector<int> seg_offset) {
    auto tmp = v;

    // Prefix sum
    for (unsigned i = 1; i < v.size(); i++) {
        v[i] += v[i-1];
    }

    std::vector<std::thread> threads;
    for (unsigned i = 0; i < v.size(); i++) {
        threads.push_back(std::move(std::thread([&](int idx){
            v[idx] = v[idx + seg_offset[idx]] - v[idx] + tmp[idx];
        }, i)));
    }
    for (auto& t : threads) t.join();
}

#endif // UTILS_H

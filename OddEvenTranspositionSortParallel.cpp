#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h> 

void oddEvenTranspositionSort(std::vector<int>& arr) {
    int n = arr.size();
    bool sorted = false;

    while (!sorted) {
        sorted = true;

        // Perform odd indexed passes in parallel
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
            }
        }

        // Perform even indexed passes in parallel
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
            }
        }
    }
}

int main() {
    // Example array
    std::vector<int> arr = {34, 7, 23, 32, 5, 62};

    std::cout << "Original array: ";
    for (int num : arr)
        std::cout << num << " ";
    std::cout << std::endl;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Start the odd-even transposition sort
    oddEvenTranspositionSort(arr);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Sorted array: ";
    for (int num : arr)
        std::cout << num << " ";
    std::cout << std::endl;

    // Output execution time
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}

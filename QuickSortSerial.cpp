#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

void quicksort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int pivot = arr[right]; // Choose the rightmost element as pivot
        int i = left - 1;

        for (int j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                i++;
                std::swap(arr[i], arr[j]); // Swap elements
            }
        }
        std::swap(arr[i + 1], arr[right]); // Place pivot in the correct position
        int pivotIndex = i + 1;

        quicksort(arr, left, pivotIndex - 1); // Recursively sort left part
        quicksort(arr, pivotIndex + 1, right); // Recursively sort right part
    }
}

int main() {
    const int SIZE = 1000000;
    std::vector<int> arr(SIZE);

    // Generate random data
    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < SIZE; i++) {
        arr[i] = std::rand() % 1000000; // Random numbers between 0 and 999999
    }

    // Measure execution time
    std::clock_t startTime = std::clock();
    
    quicksort(arr, 0, arr.size() - 1);
    
    std::clock_t endTime = std::clock();

    std::cout << "Sorted array (first 10 elements): ";
    for (int i = 0; i < 10 && i < SIZE; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";

    double executionTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    std::cout << "Execution time: " << executionTime << " seconds" << std::endl;

    return 0;
}

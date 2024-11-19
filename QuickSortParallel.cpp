#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <ctime>

void quicksort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int pivot = arr[right];
        int i = left - 1;

        for (int j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                i++;
                std::swap(arr[i], arr[j]);
            }
        }
        std::swap(arr[i + 1], arr[right]);
        int pivotIndex = i + 1;

        #pragma omp task shared(arr)
        quicksort(arr, left, pivotIndex - 1);
        
        #pragma omp task shared(arr)
        quicksort(arr, pivotIndex + 1, right);
        
        #pragma omp taskwait
    }
}

void parallelQuicksort(std::vector<int>& arr) {
    #pragma omp parallel
    {
        #pragma omp single
        quicksort(arr, 0, arr.size() - 1);
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
    double startTime = omp_get_wtime();
    
    parallelQuicksort(arr);
    
    double endTime = omp_get_wtime();

    std::cout << "Sorted array (first 10 elements): ";
    for (int i = 0; i < 10 && i < SIZE; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Execution time: " << (endTime - startTime) << " seconds" << std::endl;

    return 0;
}

#include <iostream>
#include <omp.h>

int main() {
    int a = 10; // This will be used with firstprivate

    std::cout << "Initial value of a: " << a << std::endl;

    #pragma omp parallel for firstprivate(a)
    for (int i = 0; i < 5; ++i) {
        // Each thread has its own private copy of 'a', initialized with the original value (10)
        a += i;
        std::cout << "Thread " << omp_get_thread_num() << ": a = " << a << std::endl;
    }

    // The value of 'a' outside the parallel region remains unchanged
    std::cout << "Final value of a: " << a << std::endl;

    return 0;
}


#include <iostream>
#include <omp.h>

int main() {
    int b = 0; // This will be used with lastprivate

    std::cout << "Initial value of b: " << b << std::endl;

    #pragma omp parallel for lastprivate(b)
    for (int i = 0; i < 5; ++i) {
        b = i; // 'b' is updated in each iteration
        std::cout << "Thread " << omp_get_thread_num() << ": b = " << b << std::endl;
    }

    // The value of 'b' after the loop will be from the last iteration
    std::cout << "Final value of b: " << b << std::endl;

    return 0;
}

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <omp.h>

double monte_carlo_pi_parallel(int num_samples, int num_threads) {
    int inside_circle = 0;

    // Set the number of threads to be used in parallel
    omp_set_num_threads(num_threads);

    // Parallelize the loop using OpenMP
    #pragma omp parallel
    {
        // Each thread has its own private count of points inside the circle
        int thread_inside_circle = 0;
        unsigned int seed = static_cast<unsigned int>(time(0)) ^ omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < num_samples; ++i) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            if (x * x + y * y <= 1.0) {
                thread_inside_circle++;
            }
        }

        // Combine the results from each thread
        #pragma omp atomic
        inside_circle += thread_inside_circle;
    }

    // Calculate and return the estimate of pi
    return (4.0 * inside_circle) / num_samples;
}

int main() {
    int num_samples = 10000000; // Total number of random samples
    int num_threads = 4;        // Number of threads for parallel execution

    double pi_estimate = monte_carlo_pi_parallel(num_samples, num_threads);
    std::cout << "Estimated value of Pi: " << pi_estimate << std::endl;

    return 0;
}

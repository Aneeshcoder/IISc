#include <iostream>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>

using namespace std;

void parallel_prefix_sum(int* arr, int n, int num_threads) {
    omp_set_num_threads(num_threads);

    vector<int> thread_sums(num_threads, 0);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int chunk_size = (n + num_threads - 1) / num_threads;
        int start = thread_id * chunk_size;
        int end = std::min(start + chunk_size, n);

        if (start < n) {
            for (int i = start + 1; i < end; ++i) {
                arr[i] += arr[i - 1];
            }

            thread_sums[thread_id] = arr[end - 1];
        }
    }

    for (int i = 1; i < num_threads; ++i) {
        thread_sums[i] += thread_sums[i - 1];
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        if (thread_id > 0) {
            int chunk_size = (n + num_threads - 1) / num_threads;
            int start = thread_id * chunk_size;
            int end = std::min(start + chunk_size, n);
            int offset = thread_sums[thread_id - 1];

            for (int i = start; i < end; ++i) {
                arr[i] += offset;
            }
        }
    }
}

int main() {
    long array_sizes[] = {30000};  
    int thread_numbers[] = {32};

    ofstream outFile("parallel_output.csv");
    outFile << "Array Size,Threads,Time" << endl;

    for (long array_size : array_sizes) {
        for (int threads : thread_numbers) {
            
            const char* filename = "openmp/input_30k.bin";
            int array[array_size];
            size_t size = sizeof(array) / sizeof(array[0]);
            std::ifstream infile(filename, std::ios::binary);
            if (!infile) {
                std::cerr << "Error opening file for reading\n";
                return 1;
            }
            infile.read(reinterpret_cast<char*>(array), size * sizeof(int));
            infile.close();
            double total_time = 0.0;

            for (int iter = 0; iter < 1; ++iter) {
                auto start = chrono::high_resolution_clock::now();
                parallel_prefix_sum(array, array_size, threads);
                auto end = chrono::high_resolution_clock::now();

                chrono::duration<double> duration = end - start;
                total_time += duration.count();

                for (int i = 0; i < array_size; i++)
                    cout << array[i] << " ";
                cout << endl;
            }

            double avg_time = total_time;
            outFile << array_size << ',' << threads << ',' << avg_time << endl;
            delete[] array;
        }
    }
    outFile.close();
    return 0;
}
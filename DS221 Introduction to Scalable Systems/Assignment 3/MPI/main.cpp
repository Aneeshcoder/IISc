#include<iostream>
#include<vector>
#include<mpi.h>
#include<vector_io.h>

using namespace std;

void parallelSearchAndTiming(int size, int num_procs, int max_val, int num_trials) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Calculate chunk sizes and remainder
    int chunk_size = size / num_procs;
    int remainder = size % num_procs;

    // Local array to store the chunk for each process
    vector<int> local_array(chunk_size + (rank < remainder ? 1 : 0));

    // Function to generate a random array (done by rank 0)
    auto generateRandomArray = [](vector<int>& arr, int n, int max_val) {
        for (int i = 0; i < n; ++i) {
            arr[i] = rand() % max_val + 1;
        }
    };

    // Function to distribute the array to all processes
    if (rank == 0) {
        vector<int> array(size);
        generateRandomArray(array, size, max_val);

        for (int p = 0, start = 0; p < num_procs; ++p) {
            int send_count = (p < remainder) ? chunk_size + 1 : chunk_size;
            if (p == 0) {
                // Keep the first chunk for rank 0
                copy(array.begin(), array.begin() + send_count, local_array.begin());
            } else {
                // Send chunks to other ranks
                MPI_Send(array.data() + start, send_count, MPI_INT, p, 0, MPI_COMM_WORLD);
            }
            start += send_count;
        }
    } else {
        // Receive the local array chunk
        MPI_Recv(local_array.data(), local_array.size(), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Local search function for each process to search within its chunk
    auto localSearch = [](const vector<int>& local_array, int target) -> int {
        for (int i = 0; i < local_array.size(); ++i) {
            if (local_array[i] == target) {
                return i;
            }
        }
        return -1;
    };

    // Function to calculate global index from local index
    auto calculateGlobalIndex = [](int rank, int chunk_size, int remainder, int local_index) -> int {
        int local_start_index = (rank < remainder) ? rank * (chunk_size + 1) : rank * chunk_size + remainder;
        return local_start_index + local_index;
    };

    // Perform trials for searching
    double total_time = 0.0;

    for (int trial = 0; trial < num_trials; ++trial) {
        int target;

        // Rank 0 generates the target and broadcasts it to all processes
        if (rank == 0) {
            target = rand() % max_val + 1;
            // cout << "Trial " << trial + 1 << ": Searching for " << target << "\n";
        }
        MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Start measuring time for this trial
        double start_time = MPI_Wtime();
        
        // Perform local search on the chunk
        int local_found = localSearch(local_array, target);
        bool found = (local_found != -1);
        
        // We only need to use MPI when a result is found
        int global_found;
        if (found) {
            global_found = rank;
        }

        // Use MPI_Bcast to let all processes know who found the element (if any)
        MPI_Bcast(&global_found, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int global_index = -1;
        if (global_found != -1 && rank == global_found) {
            global_index = calculateGlobalIndex(rank, chunk_size, remainder, local_found);
        }

        // End the trial time measurement
        double end_time = MPI_Wtime();
        total_time += end_time - start_time;

        // Optionally, print the result for this trial
        // if (rank == 0) {
        //     if (global_index != -1) {
        //         cout << "Found " << target << " at index " << global_index << " in array.\n";
        //     } else {
        //         cout << "Target " << target << " not found in array.\n";
        //     }
        // }
    }

    // Rank 0 calculates and prints the average execution time
    if (rank == 0) {
        double avg_time = total_time / num_trials;
        cout << "Average time over " << num_trials << " trials: " << avg_time << " seconds.\n";
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int array_size = 1000000;
    int max_val = 5000000;
    int num_trials = 20;

    // number of processes from mpiexec -n $num_processes ./main
    int num_procs;

    // Call the combined parallel search and timing function
    parallelSearchAndTiming(array_size, num_procs, max_val, num_trials);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

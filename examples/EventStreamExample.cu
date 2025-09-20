#include <cuweaver/EventStreamOps.cuh>
#include <iostream>
#include <cassert>

__global__ void kernel(int *data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = value;
}

int main() {
    const int dataSize = 1024;
    int *d_data;

    // Allocate device memory
    cudaMalloc(&d_data, dataSize * sizeof(int));

    // Initialize cuweaver stream and events
    cuweaver::cudaStream stream;
    cuweaver::cudaEvent start;
    cuweaver::cudaEvent stop;

    // Record the start event
    cuweaver::eventRecord(start, stream);

    // Launch the kernel in the stream
    kernel<<<dataSize / 256, 256, 0, stream.nativeHandle()>>>(d_data, 42);

    // Record the stop event
    cuweaver::eventRecord(stop, stream);

    // Synchronize the stream to ensure the kernel has finished
    cuweaver::streamSynchronize(stream);

    // Output the elapsed time for the kernel execution
    std::cout << cuweaver::eventElapsedTime(start, stop) << " ms elapsed for kernel execution." << std::endl;

    // Clean up resources
    cudaFree(d_data);

    return 0;
}

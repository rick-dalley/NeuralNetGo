## Performance Insights and Optimization Journey

While running the neural network, I observed that the training process in Go was taking significantly longer than an equivalent implementation in C++. To identify the bottleneck, I used the Go profiler (pprof) and analyzed the results.

## Profiler Findings

The profiler revealed that the primary bottleneck was the matrix multiplication operation implemented in the Matrix.Multiply method. This method accounted for the vast majority of the execution time.

## Unoptimized

neural-net % go tool pprof cpu.prof
File: \_\_debug_bin3486075173
Type: cpu
Time: Jan 14, 2025 at 1:56pm (MST)
Duration: 225.37s, Total samples = 580.31s (257.50%)
Entering interactive mode (type "help" for commands, "o" for options)
(pprof) top
Showing nodes accounting for 547.23s, 94.30% of 580.31s total
Dropped 279 nodes (cum <= 2.90s)
Showing top 10 nodes out of 44
flat flat% sum% cum cum%
351.51s 60.57% 60.57% 351.52s 60.57% runtime.usleep
120.77s 20.81% 81.38% 120.81s 20.82% runtime.pthread_cond_wait
39.42s 6.79% 88.18% 145.69s 25.11% neural-net/matrix.(*Matrix).Multiply.func1
25.49s 4.39% 92.57% 25.49s 4.39% runtime.pthread_cond_signal
2.57s 0.44% 93.01% 136.65s 23.55% runtime.lock2
1.92s 0.33% 93.34% 6.72s 1.16% runtime.mallocgc
1.84s 0.32% 93.66% 3.74s 0.64% sync.(*WaitGroup).Add
1.32s 0.23% 93.89% 4.71s 0.81% runtime.unlock2
1.20s 0.21% 94.09% 8.26s 1.42% runtime.newproc1
1.19s 0.21% 94.30% 49.57s 8.54% neural-net/matrix.(\*Matrix).Multiply

## Optimization Attempts

    1.	Making Matrix Multiplication Concurrent:
    •	I attempted to parallelize the multiplication operation by processing rows concurrently.
    •	Result: While this approach utilized Go’s concurrency, the overhead of goroutines and synchronization far outweighed the benefits, as the method’s scope was too small to justify the added complexity.

## Using concurrency

neural-net % go tool pprof cpu.prof
File: \_\_debug_bin2744135718
Type: cpu
Time: Jan 14, 2025 at 2:05pm (MST)
Duration: 379.63s, Total samples = 327.98s (86.39%)
Entering interactive mode (type "help" for commands, "o" for options)
(pprof) top
Showing nodes accounting for 322.86s, 98.44% of 327.98s total
Dropped 121 nodes (cum <= 1.64s)
flat flat% sum% cum cum%
322.32s 98.27% 98.27% 322.45s 98.31% neural-net/matrix.(*Matrix).Multiply
0.41s 0.13% 98.40% 1.72s 0.52% neural-net/activationFunctions.ApplyNew
0.13s 0.04% 98.44% 1.81s 0.55% runtime.scanobject
0 0% 98.44% 324.14s 98.83% main.(*NeuralNetwork).forwardPass
0 0% 98.44% 325.43s 99.22% main.main
0 0% 98.44% 1.81s 0.55% runtime.gcBgMarkWorker
0 0% 98.44% 1.82s 0.55% runtime.gcBgMarkWorker.func2
0 0% 98.44% 1.82s 0.55% runtime.gcDrain
0 0% 98.44% 325.43s 99.22% runtime.main
0 0% 98.44% 2.44s 0.74% runtime.systemstack

    2.	Integrating the gonum Library:
    •	I replaced the custom multiplication implementation with the gonum library, a highly optimized Go library for numerical computations.
    •	Two helper functions were added to:
    •	Flatten the Matrix data for use with gonum.
    •	Rehydrate the gonum result back into a custom Matrix format.
    •	Result: The gonum library drastically improved performance, as it leverages optimized low-level operations for matrix math.

Profiler Results

Below is a summary of the profiling results before and after optimization:

## Using gonum

neural-net % go tool pprof cpu.prof
File: \_\_debug_bin1166781284
Type: cpu
Time: Jan 14, 2025 at 2:17pm (MST)
Duration: 22.66s, Total samples = 68.47s (302.20%)
Entering interactive mode (type "help" for commands, "o" for options)
(pprof) top
Showing nodes accounting for 65.19s, 95.21% of 68.47s total
Dropped 150 nodes (cum <= 0.34s)
Showing top 10 nodes out of 55
flat flat% sum% cum cum%
47.50s 69.37% 69.37% 47.50s 69.37% gonum.org/v1/gonum/internal/asm/f64.AxpyUnitary
10.64s 15.54% 84.91% 58.14s 84.91% gonum.org/v1/gonum/blas/gonum.dgemmSerialNotNot
1.77s 2.59% 87.50% 1.77s 2.59% runtime.usleep
1.39s 2.03% 89.53% 1.39s 2.03% runtime.(*mspan).typePointersOfUnchecked
0.89s 1.30% 90.83% 0.89s 1.30% runtime.madvise
0.79s 1.15% 91.98% 0.79s 1.15% math.archExp
0.64s 0.93% 92.92% 1.64s 2.40% neural-net/activationFunctions.ApplyNew
0.60s 0.88% 93.79% 0.91s 1.33% encoding/csv.(*Reader).readRecord
0.54s 0.79% 94.58% 1.48s 2.16% main.LoadMNIST
0.43s 0.63% 95.21% 0.43s 0.63% runtime.memmove

Conclusion

Switching to the gonum library provided a significant boost in performance and reduced the complexity of the code. To help others understand the trade-offs, I’ve left the original Matrix.Multiply implementations commented out in the code. These alternatives illustrate the evolution of the solution and highlight the benefits of using optimized libraries like gonum.

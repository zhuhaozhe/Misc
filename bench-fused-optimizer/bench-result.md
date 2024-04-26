| Parameters | Parameter Size | `compiled_single_tensor_adam` (ms/iter) | `_single_tensor_adam` (ms/iter) | `_fused_adam` (ms/iter) |
|------------|----------------|------------------------------------------|----------------------------------|-------------------------|
| 1          | 16384          | 0.2226                                   | 3.0747                           | 2.0181                  |
| 2          | 16384          | 0.2464                                   | 3.0296                           | 0.9550                  |
| 4          | 16384          | 0.3280                                   | 2.4210                           | 1.0760                  |
| 8          | 16384          | 0.4870                                   | 3.6172                           | 1.2503                  |
| 16         | 16384          | 1.2157                                   | 6.4759                           | 1.7313                  |
| 32         | 16384          | 2.7263                                   | 12.0050                          | 2.3096                  |
| 1          | 65536          | 0.2958                                   | 1.4892                           | 0.8374                  |
| 2          | 65536          | 0.4340                                   | 2.1441                           | 1.0390                  |
| 4          | 65536          | 0.6854                                   | 3.5902                           | 1.1051                  |
| 8          | 65536          | 1.3468                                   | 6.1781                           | 1.2786                  |
| 16         | 65536          | 3.9141                                   | 11.4990                          | 1.5761                  |
| 32         | 65536          | 15.7267                                  | 22.5567                          | 2.3297                  |
| 1          | 262144         | 0.6532                                   | 1.4382                           | 0.8449                  |
| 2          | 262144         | 1.1461                                   | 2.2305                           | 0.9359                  |
| 4          | 262144         | 2.1010                                   | 3.6159                           | 1.0567                  |
| 8          | 262144         | 4.6804                                   | 6.6111                           | 1.5134                  |
| 16         | 262144         | 15.9823                                  | 12.8755                          | 2.5378                  |
| 32         | 262144         | 39.1214                                  | 24.8270                          | 4.2691                  |
| 1          | 1048576        | 5.4086                                   | 1.7953                           | 0.8574                  |
| 2          | 1048576        | 3.0704                                   | 2.9084                           | 1.3554                  |
| 4          | 1048576        | 2.7214                                   | 5.4979                           | 1.8966                  |
| 8          | 1048576        | 4.4198                                   | 11.2234                          | 3.2270                  |
| 16         | 1048576        | 10.9654                                  | 22.4123                          | 5.5960                  |
| 32         | 1048576        | 24.7420                                  | 44.7208                          | 10.5392                 |
| 1          | 4194304        | 6.8618                                   | 2.6646                           | 1.6738                  |
| 2          | 4194304        | 4.3225                                   | 5.2991                           | 2.9045                  |
| 4          | 4194304        | 6.5019                                   | 9.6921                           | 4.7629                  |
| 8          | 4194304        | 13.0058                                  | 19.2961                          | 9.3567                  |
| 16         | 4194304        | 36.6758                                  | 38.2443                          | 17.6571                 |
| 32         | 4194304        | 132.9222                                 | 77.1480                          | 35.4720                 |



From chatGPT:

### Performance Analysis of `compiled_single_tensor_adam`

#### Good Performance:
- When the number of parameters is small (e.g., 1 or 2) and the parameter size is relatively small (e.g., 16384), `compiled_single_tensor_adam` tends to have good performance compared to the other methods. For example, at 1 parameter and 16384 size, `compiled_single_tensor_adam` performs the best.
- For larger parameter sizes (e.g., 65536 or 262144), `compiled_single_tensor_adam` still shows good performance, especially when the number of parameters is small.
- Generally, `compiled_single_tensor_adam` performs well when dealing with smaller datasets or models.

#### Poor Performance:
- As the number of parameters increases, especially beyond 16, and the parameter size becomes larger, the performance of `compiled_single_tensor_adam` relative to the other methods deteriorates.
- At higher parameter sizes (e.g., 1048576 or 4194304), `compiled_single_tensor_adam` tends to have poorer performance, even with a smaller number of parameters.

#### Factors Affecting Performance:
- The performance of `compiled_single_tensor_adam` is likely influenced by factors such as memory bandwidth, cache size, and CPU architecture.
- Larger parameter sizes may lead to increased memory access times and cache misses, impacting the performance of compiled operations.
- The overhead of compiling operations may become significant when dealing with larger parameter sizes and numbers, leading to relatively poorer performance compared to the non-compiled versions.

#### Recommendation:
- For small models or datasets, especially with fewer parameters, `compiled_single_tensor_adam` can provide performance benefits.
- However, for larger models with many parameters, it's essential to consider the trade-offs between the compilation overhead and the actual computation time. In such cases, the non-compiled versions (`_single_tensor_adam` and `_fused_adam`) might offer better overall performance.

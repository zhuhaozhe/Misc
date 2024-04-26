| Num of Params | Parameter Size | compiled_single_tensor_adam | _single_tensor_adam | _fused_adam |
|---------------|----------------|------------------------------|---------------------|-------------|
| 1             | 16384          | 0.2034 ms/iter               | 0.2106 ms/iter      | 0.1361 ms/iter |
| 2             | 16384          | 0.2251 ms/iter               | 0.2996 ms/iter      | 0.1332 ms/iter |
| 4             | 16384          | 0.3012 ms/iter               | 0.4696 ms/iter      | 0.1611 ms/iter |
| 8             | 16384          | 0.4846 ms/iter               | 0.8218 ms/iter      | 0.2031 ms/iter |
| 16            | 16384          | 1.1454 ms/iter               | 1.5392 ms/iter      | 0.2868 ms/iter |
| 32            | 16384          | 4.3181 ms/iter               | 2.9395 ms/iter      | 0.4917 ms/iter |
| 1             | 65536          | 0.2841 ms/iter               | 0.2746 ms/iter      | 0.1390 ms/iter |
| 2             | 65536          | 0.4124 ms/iter               | 0.4423 ms/iter      | 0.1520 ms/iter |
| 4             | 65536          | 0.6579 ms/iter               | 0.7844 ms/iter      | 0.1984 ms/iter |
| 8             | 65536          | 1.3377 ms/iter               | 1.4484 ms/iter      | 0.2725 ms/iter |
| 16            | 65536          | 3.9601 ms/iter               | 2.7810 ms/iter      | 0.4253 ms/iter |
| 32            | 65536          | 14.8106 ms/iter              | 5.4129 ms/iter      | 0.7373 ms/iter |
| 1             | 262144         | 0.1995 ms/iter               | 0.2947 ms/iter      | 0.1692 ms/iter |
| 2             | 262144         | 0.2697 ms/iter               | 0.4952 ms/iter      | 0.2268 ms/iter |
| 4             | 262144         | 0.4089 ms/iter               | 0.8743 ms/iter      | 0.3372 ms/iter |
| 8             | 262144         | 0.7278 ms/iter               | 1.6053 ms/iter      | 0.5659 ms/iter |
| 16            | 262144         | 1.8102 ms/iter               | 3.1584 ms/iter      | 1.0801 ms/iter |
| 32            | 262144         | 4.7531 ms/iter               | 6.4044 ms/iter      | 2.1426 ms/iter |
| 1             | 1048576        | 0.3672 ms/iter               | 0.4418 ms/iter      | 0.3020 ms/iter |
| 2             | 1048576        | 0.6299 ms/iter               | 0.7480 ms/iter      | 0.5131 ms/iter |
| 4             | 1048576        | 1.2228 ms/iter               | 1.4078 ms/iter      | 0.9820 ms/iter |
| 8             | 1048576        | 2.5887 ms/iter               | 2.7965 ms/iter      | 1.9518 ms/iter |
| 16            | 1048576        | 7.9112 ms/iter               | 5.7584 ms/iter      | 3.9523 ms/iter |
| 32            | 1048576        | 18.0523 ms/iter              | 11.4068 ms/iter     | 8.0193 ms/iter |
| 1             | 4194304        | 1.1879 ms/iter               | 1.6175 ms/iter      | 0.9577 ms/iter |
| 2             | 4194304        | 2.3574 ms/iter               | 3.5431 ms/iter      | 1.9173 ms/iter |
| 4             | 4194304        | 4.7457 ms/iter               | 7.0853 ms/iter      | 3.8658 ms/iter |
| 8             | 4194304        | 10.9819 ms/iter              | 14.0519 ms/iter     | 7.8321 ms/iter |
| 16            | 4194304        | 32.6533 ms/iter              | 28.1309 ms/iter     | 15.8077 ms/iter |
| 32            | 4194304        | 107.3979 ms/iter             | 56.0444 ms/iter     | 31.7812 ms/iter |



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

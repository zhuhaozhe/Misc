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

### When `compiled_single_tensor_adam` performs well:

- **Small to Medium-Sized Models**:
  - `compiled_single_tensor_adam` tends to perform well with small to medium-sized models where the computational overhead is manageable. In these cases, the benefits of the compiled optimization outweigh any potential drawbacks.

- **Low to Moderate Parameter Counts**:
  - When the number of parameters is low to moderate, `compiled_single_tensor_adam` can efficiently handle the optimization process without significant overhead. This is especially true when the parameter count doesn't result in excessive memory usage or computational complexity.

- **Regularly Executed Code Paths**:
  - If the code paths optimized by `compiled_single_tensor_adam` are frequently executed, the upfront compilation cost can be amortized over multiple invocations, leading to improved performance over time.

### When `compiled_single_tensor_adam` may struggle:

- **Large Models with High Parameter Counts**:
  - `compiled_single_tensor_adam` might struggle with large models containing a high number of parameters. The overhead of compiling the optimization process for such models could outweigh the benefits, leading to longer initialization times and potentially suboptimal performance during runtime.

- **Infrequently Executed Code Paths**:
  - If the code paths optimized by `compiled_single_tensor_adam` are rarely executed, the upfront compilation cost may not be justified, and the performance benefits might not be realized. In such cases, the overhead of compilation could outweigh any gains from optimization.

- **Dynamic or Changing Model Architectures**:
  - Models with dynamic or frequently changing architectures may not be well-suited for `compiled_single_tensor_adam`. The upfront compilation process assumes a fixed model structure, making it less adaptable to dynamic changes during runtime.

- **Limited Memory Resources**:
  - If the system has limited memory resources, the additional memory required for the compiled optimization process could strain the system, leading to performance degradation or even out-of-memory errors.

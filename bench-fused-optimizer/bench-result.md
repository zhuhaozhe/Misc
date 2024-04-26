| Num Params | Param Size | compiled_single_tensor_adam | _single_tensor_adam | _fused_adam |
|------------|------------|-----------------------------|---------------------|-------------|
| 1          | 16384      | 0.2135 ms/iter              | 2.4358 ms/iter      | 1.2023 ms/iter |
| 2          | 16384      | 0.2503 ms/iter              | 2.4695 ms/iter      | 0.8987 ms/iter |
| 4          | 16384      | 0.3161 ms/iter              | 2.7614 ms/iter      | 0.9782 ms/iter |
| 8          | 16384      | 0.4922 ms/iter              | 3.6344 ms/iter      | 1.2391 ms/iter |
| 16         | 16384      | 1.2241 ms/iter              | 6.5016 ms/iter      | 1.6002 ms/iter |
| 32         | 16384      | 2.7186 ms/iter              | 11.8762 ms/iter     | 2.2159 ms/iter |
| 1          | 65536      | 0.2983 ms/iter              | 1.4001 ms/iter      | 0.9435 ms/iter |
| 2          | 65536      | 0.4314 ms/iter              | 2.0828 ms/iter      | 0.9834 ms/iter |
| 4          | 65536      | 0.6840 ms/iter              | 3.5213 ms/iter      | 1.0578 ms/iter |
| 8          | 65536      | 1.3470 ms/iter              | 6.2054 ms/iter      | 1.3649 ms/iter |
| 16         | 65536      | 3.9092 ms/iter              | 11.4269 ms/iter     | 1.5024 ms/iter |
| 32         | 65536      | 15.7774 ms/iter             | 22.2950 ms/iter     | 2.2080 ms/iter |
| 1          | 262144     | 0.6622 ms/iter              | 1.4159 ms/iter      | 0.9140 ms/iter |
| 2          | 262144     | 1.1475 ms/iter              | 2.1536 ms/iter      | 0.9618 ms/iter |
| 4          | 262144     | 2.1019 ms/iter              | 3.6088 ms/iter      | 1.0419 ms/iter |
| 8          | 262144     | 4.7946 ms/iter              | 6.5341 ms/iter      | 1.5328 ms/iter |
| 16         | 262144     | 15.7025 ms/iter             | 12.8001 ms/iter     | 2.5711 ms/iter |
| 32         | 262144     | 39.0621 ms/iter             | 25.2774 ms/iter     | 4.0783 ms/iter |
| 1          | 1048576    | 3.0055 ms/iter              | 1.7499 ms/iter      | 0.7868 ms/iter |
| 2          | 1048576    | 2.2222 ms/iter              | 2.8946 ms/iter      | 1.3644 ms/iter |
| 4          | 1048576    | 2.9018 ms/iter              | 5.4443 ms/iter      | 2.1006 ms/iter |
| 8          | 1048576    | 4.3899 ms/iter              | 11.1489 ms/iter     | 3.1717 ms/iter |
| 16         | 1048576    | 11.0814 ms/iter             | 22.5518 ms/iter     | 5.8045 ms/iter |
| 32         | 1048576    | 24.7874 ms/iter             | 44.7617 ms/iter     | 10.5161 ms/iter |
| 1          | 4194304    | 3.9647 ms/iter              | 2.6449 ms/iter      | 1.6035 ms/iter |
| 2          | 4194304    | 4.1700 ms/iter              | 5.2296 ms/iter      | 2.6644 ms/iter |
| 4          | 4194304    | 6.5485 ms/iter              | 9.9678 ms/iter      | 4.9528 ms/iter |
| 8          | 4194304    | 13.0899 ms/iter             | 19.5815 ms/iter     | 9.3918 ms/iter |
| 16         | 4194304    | 36.7455 ms/iter             | 38.7984 ms/iter     | 17.8791 ms/iter |
| 32         | 4194304    | 129.0258 ms/iter            | 77.6986 ms/iter     | 35.3970 ms/iter |
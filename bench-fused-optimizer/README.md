Using https://github.com/pytorch/pytorch/tree/gh/zhuhaozhe/18/orig

bash bench_opt.sh

My test result on 32 core ICX
output:
```
small bench on 1 thread
TENSOR_SIZE=262144 NPARAM=4 numactl -C 1 -m 0 python compare_adam.py
compiled_single_tensor_adam time: 5.2002 seconds
_single_tensor_adam time: 23.5194 seconds
_fused_adam time: 7.2299 seconds
large bench on 1 numa-node
TENSOR_SIZE=4194304 NPARAM=32 numactl -C 0-31 -m 0 python compare_adam.py
compiled_single_tensor_adam time: 4.6700 seconds
_single_tensor_adam time: 3.0243 seconds
_fused_adam time: 0.9501 seconds
```

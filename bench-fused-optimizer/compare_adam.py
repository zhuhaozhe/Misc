from torch.optim.adam import _single_tensor_adam, _fused_adam
import torch
import copy
device='cpu'
dtype=torch.float
import os

TENSOR_SIZE = (int(os.getenv('ELEMENT_SIZE_PER_PARAM', 512 * 512)), )  # make sure to cover cpu vec and scalar
NPARAM = int(os.getenv("NPARAM", 4))

kwargs = {}
kwargs['params'] = [torch.randn(TENSOR_SIZE, device=device, dtype=dtype) for _ in range(NPARAM)]
kwargs['grads'] = [torch.randn(TENSOR_SIZE, device=device, dtype=dtype) for _ in range(NPARAM)]
kwargs['exp_avgs'] = [torch.randn(TENSOR_SIZE, device=device, dtype=dtype) for _ in range(NPARAM)]
kwargs['exp_avg_sqs'] = [torch.randn(TENSOR_SIZE, device=device, dtype=dtype) for _ in range(NPARAM)]
kwargs['state_steps'] = [torch.tensor([10.0], device=device) for _ in range(NPARAM)]
kwargs['grad_scale'] = None
kwargs['found_inf'] = None
kwargs['beta1'] = 0.9
kwargs['beta2'] = 0.999
kwargs['lr'] = 0.1
kwargs['eps'] = 1e-8
kwargs['has_complex'] = False
kwargs['capturable'] = False
kwargs['differentiable'] = False
kwargs['amsgrad'] = True
kwargs['maximize'] = False
kwargs['weight_decay'] = 0.01
kwargs['max_exp_avg_sqs'] = [torch.randn(TENSOR_SIZE, device=device, dtype=dtype) for _ in range(NPARAM)]

kwargs_a = copy.deepcopy(kwargs)
kwargs_b = copy.deepcopy(kwargs)
kwargs_c = copy.deepcopy(kwargs)

compile_adam = torch.compile(_single_tensor_adam)
compile_adam(**kwargs)

a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
def cache_flush():
    # We assume the cache size is <= 512MB here.
    # a = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
    # b = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float)
    # a, b are initialized out of this function to avoid allocate memory every time
    global a, b
    a += b

import time
def bench(fn, kwargs, warmup=100, bench_iters=100):
    for _ in range(warmup):
        cache_flush()
        fn(**kwargs)
    start_time = time.time()
    for _ in range(bench_iters):
        cache_flush()
        fn(**kwargs)
    end_time = time.time()
    print(f"{fn.__name__} time: {end_time - start_time:.4f} seconds")
compile_adam.__name__ = "compiled_single_tensor_adam"
bench(compile_adam, kwargs_a)
bench(_single_tensor_adam, kwargs_b)
bench(_fused_adam, kwargs_c)
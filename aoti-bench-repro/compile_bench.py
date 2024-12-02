import os

def aoti_benchmark_compile(ninstances, nbatches, bs, tmp_dir, target_dir):
    import textwrap
    inference_template = textwrap.dedent(
        """
        #include <vector>

        #include <torch/torch.h>
        #include <torch/script.h>
        #include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

        #include <iostream>

        int main() {
            c10::InferenceMode mode;
            size_t ninstances = %s;
            if (ninstances == 0) ninstances = 1;
            size_t niters = %s;
            size_t total_iters = ninstances * niters;
            size_t bs = %s;
            auto module = torch::jit::load("%s");
            std::vector<torch::Tensor> _input_vec = module.attr("tensor_list").toTensorList().vec();
            std::vector<torch::Tensor> input_vec;
            for (const auto t : _input_vec) {
                input_vec.push_back(t.clone());
            }
            torch::inductor::AOTIModelContainerRunnerCpu runner("%s", ninstances * 2);
            std::vector<torch::Tensor> outputs = runner.run(input_vec);

            using Input = std::vector<torch::Tensor>;
            std::vector<std::vector<Input>> thread_inputs(ninstances);
            std::vector<size_t> input_iters(ninstances);
            for (const auto thread_id : c10::irange(ninstances)) {
                for (const auto i [[maybe_unused]] : c10::irange(niters * 2 + 100))  {
                    thread_inputs[thread_id].push_back(input_vec);
                }
                input_iters[thread_id] = 0;
            }
            std::atomic<int64_t> num_attempted_iters{0};
            std::mutex m;
            std::condition_variable worker_main_cv;
            std::condition_variable main_worker_cv;
            int64_t initialized{0};
            int64_t finished{0};
            bool start{false};
            std::vector<std::thread> callers;
            callers.reserve(ninstances);
            std::cout << "init done, benchmark start" << std::endl;
            for (const auto thread_id : c10::irange(ninstances)) {
                callers.emplace_back([&, thread_id]() {
                    // warmup 100 iters
                    for (const auto j : c10::irange(100)) {
                        (void)j;
                        runner.run(thread_inputs[thread_id][input_iters[thread_id]]);
                        ++input_iters[thread_id];
                    }
                    {
                        std::unique_lock<std::mutex> lock(m);
                        ++initialized;
                        worker_main_cv.notify_one();
                        while (!start) {
                            main_worker_cv.wait(lock);
                        }
                    }
                    while (num_attempted_iters.fetch_add(1) < total_iters) {
                        runner.run(thread_inputs[thread_id][input_iters[thread_id]]);
                        ++input_iters[thread_id];
                    }

                    {
                        std::unique_lock<std::mutex> lock(m);
                        ++finished;
                        worker_main_cv.notify_one();
                    }
                });
            }

            using Clock = std::chrono::high_resolution_clock;
            using RecordProfile = torch::autograd::profiler::RecordProfile;
            using TimePoint = std::chrono::time_point<Clock>;
            TimePoint start_time;
            {
                std::unique_lock<std::mutex> lock(m);
                while (initialized != ninstances) {
                    worker_main_cv.wait(lock);
                }
                start = true;
                start_time = Clock::now();
            }
            main_worker_cv.notify_all();
            {
                std::unique_lock<std::mutex> lock(m);
                worker_main_cv.wait(
                    lock, [&]() { return finished == ninstances; });
            }
            auto end_time = std::chrono::high_resolution_clock::now();

            float total_time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        end_time - start_time)
                                        .count() / 1000.0 / 1000.0;
            float fps = bs * ninstances * niters / total_time_ms * 1000;
            std::cout << "Throughput: " << fps << std::endl;
            for (auto& t : callers) {
                t.join();
            }
            return 0;
        }
        """
    )
    # os.system(f"cp {tmp_dir}/model.so {target_dir}/model.so")
    os.system(f"ln -s {tmp_dir}/model.so {target_dir}/model.so")
    os.system(f"cp {tmp_dir}/inputs.pt {target_dir}/inputs.pt")
    model_dir = f"{target_dir}/model.so"
    inputs_dir = f"{target_dir}/inputs.pt"
    src_code = inference_template % (
        ninstances,
        nbatches,
        bs,
        inputs_dir,
        model_dir,
    )
    with open(f"{target_dir}/bench.cpp", "w") as f:
        f.write(src_code)
    os.system(f"cp ./CMakeLists.txt {target_dir}/CMakeLists.txt")
    cmake_prefix_path = torch.utils.cmake_prefix_path
    pytorch_install_dir = os.path.dirname(os.path.abspath(torch.__file__))
    torch_libraries = os.path.join(pytorch_install_dir, "lib")
    os.system(f"export CMAKE_PREFIX_PATH={cmake_prefix_path} && export TORCH_LIBRARIES={torch_libraries} && cd {target_dir} && cmake . && make")
    return f"{target_dir}/aoti_example"

def gen_model_so(model, example_inputs):
    tmp_dir = os.getcwd()
    model_dir = f"{tmp_dir}/model.so"
    inputs_dir = f"{tmp_dir}/inputs.pt"
    torch._export.aot_compile(
        model, example_inputs,
        options={"aot_inductor.output_path":model_dir}
    )
    # save example inputs and loaded it in cpp later
    runner = torch._C._aoti.AOTIModelContainerRunnerCpu(model_dir, 1)  # type: ignore[call-arg]
    call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
    import torch.utils._pytree as pytree
    in_spec = pytree.treespec_loads(call_spec[0])
    from torch.export._tree_utils import reorder_kwargs
    flat_inputs = pytree.tree_flatten((example_inputs, reorder_kwargs({}, in_spec)))[0]
    flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
    class TensorListModule(torch.nn.Module):
        def __init__(self, tensor_list):
            super(TensorListModule, self).__init__()
            self.tensor_list = tensor_list

        def forward(self):
            return self.tensor_list

    # Create an instance of the module
    module = TensorListModule(flat_inputs)
    # Save the module
    torch.jit.save(torch.jit.script(module), inputs_dir)
    print(f"{tmp_dir}")
    return f"{tmp_dir}"


import torch
import torch.nn as nn
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

input_size = 128
hidden_size = 1024
output_size = 1024
BS = 256
def create_model():
    model = TwoLayerNet(input_size, hidden_size, output_size)
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
    from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
    from torch._export import capture_pre_autograd_graph
    with torch.no_grad():
        example_inputs = torch.randn(BS, 128)
        exported_model = capture_pre_autograd_graph(
            model,
            (example_inputs, ),
        )
        quantizer = X86InductorQuantizer()
        quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
        prepared_model = prepare_pt2e(exported_model, quantizer)
        prepared_model(example_inputs)
        converted_model = convert_pt2e(prepared_model)
        torch.ao.quantization.move_exported_model_to_eval(converted_model)
    return model


if __name__ == "__main__":
    import os
    import sys
    args = sys.argv
    nprocess = int(args[1])
    nthread = int(args[2])
    coreidx_per_numa = list(range(0, nthread * nprocess + 1, nthread))
    numas = len(coreidx_per_numa) - 1

    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["MALLOC_CONF"] = "oversize_threshold:1,background_thread:true,metadata_thp:auto"
    assert "LD_PRELOAD" in os.environ.keys()
    assert "iomp" in os.environ["LD_PRELOAD"]
    assert ("jemalloc" in os.environ["LD_PRELOAD"] or "tcmalloc" in os.environ["LD_PRELOAD"])
    model = create_model()
    model_so_dir = gen_model_so(model, (torch.randn(BS, 128), ))
    EVAL_BATCH = 1000
    composed_cmd = ""
    for i in range(numas):
        target_dir = f"./aoti-dir-{i}"
        if os.path.exists(target_dir):
            os.system(f"rm -r {target_dir}")
        os.system(f"mkdir {target_dir}")
        start = coreidx_per_numa[i]
        end = coreidx_per_numa[i + 1] - 1
        bench_bin = aoti_benchmark_compile(end-start+1, EVAL_BATCH, BS, model_so_dir, target_dir)
        composed_cmd += f"taskset -c {start}-{end} {bench_bin} "
        if i != (numas - 1):
            composed_cmd += " & "

    print(composed_cmd)
    os.system(composed_cmd)
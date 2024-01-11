import argparse
import re
import os
import time

def ArgParser():
    parser = argparse.ArgumentParser(description='OneDNN Verbose Toolkit')
    parser.add_argument('--file', '-f', type=str, default="./output_code.py")
    parser.add_argument('--kernel', '-k', type=str, default="fused_0")
    parser.add_argument('--target-dir', '-t', type=str, default="./kernel_debug")
    return parser

def save_tensors(args, target_folder):

    cpp_kernel_end = False
    def save_tensors_for_kernel(line, fp):
        tensors = []
        indent = ""
        s = 0
        while line[s] == " ":
            indent += " "
            s += 1
        pattern = re.compile(r'c_void_p\S+data_ptr')
        result  = pattern.findall(line)
        for r in result:
            # 'c_void_p(_frozen_param137.data_ptr'
            tensor = r[9:-9]
            tensors.append(tensor)
            save = f"{indent}torch.save({tensor}, \"{target_folder}/{tensor}.pt\")\n"
            fp.writelines(save)
        fp.writelines(f"{indent}exit()\n")
        return tensors

    file_name = args.file
    kernel_name = args.kernel
    fp = open(file_name, 'r')
    content = fp.readlines()
    fp.close()
    fp = open(f"{target_folder}/tmp.py", 'w')
    for line in content:
        if "del async_compile" in line:
            cpp_kernel_end = True
        if kernel_name in line and cpp_kernel_end:
            tensors = save_tensors_for_kernel(line, fp)
        fp.write(line)
    fp.close()
    os.system(f"python {target_folder}/tmp.py")
    return tensors

def generate_single_kernel_bench(args, target_folder, to_load):

    def load_tensors(tensors, fp):
        for t in tensors:
            fp.writelines(f"    {t} = torch.load(\"{target_folder}/{t}.pt\")\n")

    def generate_args_string(tensors):
        args = ""
        for t in tensors:
            args += f"c_void_p({t}.data_ptr()), "
        return args

    file_name = args.file
    kernel_name = args.kernel
    fp = open(file_name, 'r')
    content = fp.readlines()
    fp.close()
    init_lines = True
    target_kernels = False
    fp = open(f"{target_folder}/{kernel_name}.py", 'w')
    for line in content:
        if init_lines or target_kernels:
            fp.write(line)
        if "async_compile = AsyncCompile()" in line:
            init_lines = False
        if line.startswith(kernel_name):
            fp.write(line)
            target_kernels = True
        if target_kernels and line == "\n":
            target_kernels = False
            break
    fp.writelines(
r"""
async_compile.wait(globals())
del async_compile

def benchmark_compiled_module(times=10, repeat=100):
    from torch._inductor.utils import print_performance
"""
    )
    load_tensors(to_load, fp)
    args = generate_args_string(to_load)
    caller = f"    fn = lambda: {kernel_name}({args})\n" 
    fp.writelines(caller)
    fp.writelines(
r"""
    return print_performance(fn, times=times, repeat=repeat)

if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('single_kernel', benchmark_compiled_module)
"""
    )
    fp.close()
    x = os.system(f"python {target_folder}/{kernel_name}.py")
    print(f"file saved to {target_folder}/{kernel_name}.py")


def mkdir(args):
    folder = os.path.exists(args.target_dir)
    if not folder:
        os.makedirs(args.target_dir)
    timestamp = time.strftime('%Y-%m-%d::%H:%M:%S', time.localtime())
    os.makedirs(f"{args.target_dir}/{timestamp}")
    return f"{args.target_dir}/{timestamp}"
    

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    target_folder = mkdir(args)
    saved_tensors = save_tensors(args, target_folder)
    generate_single_kernel_bench(args, target_folder, saved_tensors)

export LD_PRELOAD=/localdisk1/haozhe/miniforge3/envs/dlrm/lib/libtcmalloc.so:/localdisk1/haozhe/miniforge3/envs/dlrm/lib/libiomp5.so
python compile_bench.py 6 40 2>&1 | tee 6_40.log
python compile_bench.py 5 40 2>&1 | tee 5_40.log
python compile_bench.py 4 40 2>&1 | tee 4_40.log
python compile_bench.py 3 40 2>&1 | tee 3_40.log
python compile_bench.py 2 40 2>&1 | tee 2_40.log
python compile_bench.py 1 40 2>&1 | tee 1_40.log

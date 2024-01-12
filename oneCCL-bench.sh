export ONECCL_HOME=/home/pytorch/haozhe/torch-ccl-bench/oneCCL/build/_install
source ${ONECCL_HOME}/env/setvars.sh
export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:{CONDA_PREFIX}/lib/libiomp5.so
export WRAM_UP_ITRES=${WRAM_UP_ITRES:-1000}
export BENCH_ITERS=${BENCH_ITERS:-10000}

export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-mpi}
export COLLECTIVE=${COLLECTIVE:-allreduce}

export FI_PROVIDER=psm3

#export PSM3_IDENTIFY=1
#export PSM3_TRACEMASK=0x3
#export FI_LOG_INFO=debug

export CCL_ATL_SHM=${CCL_ATL_SHM:-0}

export ELE_SIZE=${CCL_ATL_SHM:-16384}

# export CCL_LOG_LEVEL=trace
NUM_CCL_WORKER=${NUM_CCL_WORKER:-2}

if [[ $NUM_CCL_WORKER = 1 ]]
then
    I_MPI_PIN_DOMAIN=[0xfffffffe,0xfffffffe00000000]
    CCL_WORKER_AFFINITY=0,32
elif [[ $NUM_CCL_WORKER = 2 ]]
then
    I_MPI_PIN_DOMAIN=[0xfffffffc,0xfffffffc00000000]
    CCL_WORKER_AFFINITY=0,1,32,33
elif [[ $NUM_CCL_WORKER = 4 ]]
then
    I_MPI_PIN_DOMAIN=[0xfffffff0,0xfffffff000000000]
    CCL_WORKER_AFFINITY=0,1,2,3,32,33,34,35
elif [[ $NUM_CCL_WORKER = 8 ]]
then
    I_MPI_PIN_DOMAIN=[0xffffff00,0xffffff0000000000]
    CCL_WORKER_AFFINITY=0,1,2,3,4,5,6,7,32,33,34,35,36,37,38,39
fi

export CCL_WORKER_COUNT=$NUM_CCL_WORKER
OMP_NUM_THREADS=`expr 32 - $NUM_CCL_WORKER`


echo “mpiexec.hydra -np 2 --ppn 2 -f hostfile \
                        -genv KMP_AFFINITY=granularity=fine,compact,1,0 \
                        -genv KMP_BLOCKTIME=1 \
                        -genv CCL_WORKER_COUNT=$NUM_CCL_WORKER \
                        -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
                        -genv MASTER_ADDR=127.0.0.1 \
                        -genv MASTER_PORT=29500 \
                        -genv I_MPI_PIN_DOMAIN=$I_MPI_PIN_DOMAIN \
                        -genv CCL_WORKER_AFFINITY=$CCL_WORKER_AFFINITY \
                        ${ONECCL_HOME}/examples/benchmark/benchmark -l ${COLLECTIVE} -w ${WRAM_UP_ITRES} -i ${BENCH_ITERS} -y ${ELE_SIZE} -d bfloat16”

mpiexec.hydra -np 2 --ppn 2 -f hostfile \
                        -genv KMP_AFFINITY=granularity=fine,compact,1,0 \
                        -genv KMP_BLOCKTIME=1 \
                        -genv CCL_WORKER_COUNT=$NUM_CCL_WORKER \
                        -genv OMP_NUM_THREADS=$OMP_NUM_THREADS \
                        -genv MASTER_ADDR=127.0.0.1 \
                        -genv MASTER_PORT=29500 \
                        -genv I_MPI_PIN_DOMAIN=$I_MPI_PIN_DOMAIN \
                        -genv CCL_WORKER_AFFINITY=$CCL_WORKER_AFFINITY \
                        ${ONECCL_HOME}/examples/benchmark/benchmark -l ${COLLECTIVE} -w ${WRAM_UP_ITRES} -i ${BENCH_ITERS} -y ${ELE_SIZE} -d bfloat16

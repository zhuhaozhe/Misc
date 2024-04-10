export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $4}'`
SOCKETS=`lscpu | grep "Socket(s)" | awk '{print $2}'`
NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk '{print $3}'`
NUMA_NODES_PER_SOCKETS=`expr $NUMA_NODES / $SOCKETS`
CORES_PER_NUMA_NODE=`expr $CORES_PER_SOCKET / $NUMA_NODES_PER_SOCKETS`
END_CORE_ID=`expr $CORES_PER_NUMA_NODE - 1`

echo "small bench on 1 thread"
echo "TENSOR_SIZE=262144 NPARAM=4 numactl -C 1 -m 0 python compare_adam.py"
TENSOR_SIZE=262144 NPARAM=4 numactl -C 1 -m 0 python compare_adam.py
echo "large bench on 1 numa-node"
echo "TENSOR_SIZE=4194304 NPARAM=32 numactl -C 0-$END_CORE_ID -m 0 python compare_adam.py"
TENSOR_SIZE=4194304 NPARAM=32 numactl -C 0-$END_CORE_ID -m 0 python compare_adam.py

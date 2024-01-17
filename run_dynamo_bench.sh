export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

multi_threads_test() {
    CORES=$(lscpu | grep Core | awk '{print $4}')
    export OMP_NUM_THREADS=$CORES
    end_core=$(expr $CORES - 1)    
    numactl -C 0-${end_core} --membind=0 python benchmarks/dynamo/${SUITE}.py --${SCENARIO} --${DT} -dcpu -n50 --no-skip --dashboard --only "${MODEL}" ${Channels_extra} ${BS_extra} ${Shape_extra} ${Mode_extra} ${Wrapper_extra} ${Flag_extra} --timeout 9000 --backend=inductor --output=${LOG_BASE}/${SUITE}.csv
}

SCENARIO=performance
export TORCHINDUCTOR_FREEZING=1
Flag_extra="--freezing"
Mode_extra="--inference"

for DT in float32 amp
do
for suite in timm_models huggingface torchbench
do
  export SUITE=$suite
  echo $SUITE
  export LOG_BASE=`date +%m%d%H%M%S`
  export LOG_BASE=${LOG_BASE}_${DT}_${EXTRA}
  mkdir $LOG_BASE
  multi_threads_test
done
done

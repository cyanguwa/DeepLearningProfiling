#!/bin/bash
#SBATCH -J rnn1d-tf2-noxla
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -q special
#SBATCH -A m1759
#SBATCH -t 4:00:00

#activate env
module load tensorflow/gpu-2.2.0-py37
export PROFILER='cupy'

#enalbe XLA or not, 'xla' or 'noxla'
export enable_xla='noxla'

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_HOME}
export CUDA_DIR=${CUDA_HOME}

#rankspernode
rankspernode=1

#openmp  
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
sruncmd="srun -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) --cpu_bind=cores"

#create run dir
run_dir=$PWD/tf_cnn_kernels_nsight/rnn1d-tf2-noxla/$SLURM_JOBID/
mkdir -p ${run_dir}

#copy relevant files
script_dir=../python
script="rnn1d_tf2.py"
cp $script_dir/$script $run_dir/

#step in
cd ${run_dir}


#list of metrics
#metrics="sm__cycles_elapsed.avg "
metrics="sm__cycles_elapsed.avg.per_second,\
sm__cycles_elapsed.avg,\
sm__inst_executed_pipe_tensor.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum "

#export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
#export XLA_FLAGS="--xla_dump_to=$run_dir"

cells="lstm"
precs="16 32"
batch_size="32 64 128 256"
time_steps="32 64 128"
features="32 64 128"
hidden_size="16 32 64 128"

#cells="rnn lstm gru "
#precs="16 "

num_warmup=5
num_iter=1

for prec in $precs; do
    for cell in $cells; do
        for batch in $batch_size; do
            for time in $time_steps; do
                for feature in $features; do
                    for h_size in $hidden_size; do
                        input_tensor_shape=$batch" "$time" "$feature
                        echo $input_tensor_shape

                        outputstr=tf2.fp_${prec}.celltype_${cell}.input_${batch}x${time}x${feature}.nneu_${h_size}
                        #iterate over FW BW
                        for ctype in calibrate forward backward; do

                            #profile string
                            profilestring="/usr/common/software/cuda/11.0.167/bin/nv-nsight-cu-cli \
                            --profile-from-start off --metrics ${metrics} --kernel-base demangled -o \
                            ${outputstr}.pass_${ctype}.${enable_xla}"

                            if [ $enable_xla == "xla" ];then
                                ${sruncmd} ${profilestring} $(which python) -u ./$script \
                                    --input_tensor_shape ${input_tensor_shape} \
                                    --cell_type ${cell} \
                                    --n_neurons ${h_size} \
                                    --dtype float${prec} \
                                    --num_iterations ${num_iter} \
                                    --num_warmups ${num_warmup} \
                                    --enable_xla \
                                    --compute_type ${ctype}
                            else
                                ${sruncmd} ${profilestring} $(which python) -u ./$script \
                                    --input_tensor_shape ${input_tensor_shape} \
                                    --cell_type ${cell} \
                                    --n_neurons ${h_size} \
                                    --dtype float${prec} \
                                    --num_iterations ${num_iter} \
                                    --num_warmups ${num_warmup} \
                                    --compute_type ${ctype}
                            fi
                        done
                    done
                done
            done
        done
    done
done

#!/bin/bash
#SBATCH -J rnn1d-tf1-noxla
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -q special
#SBATCH -A m1759
#SBATCH -t 4:00:00

#activate env
module load tensorflow/gpu-1.15.0-py37
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
run_dir=$PWD/tf_cnn_kernels_nsight/rnn1d-tf1-noxla/$SLURM_JOBID/
mkdir -p ${run_dir}

#copy relevant files
script_dir=../python
script="rnn1d_tf1.py"
cp $script_dir/$script $run_dir/

#step in
cd ${run_dir}

if [ $enable_xla == "xla" ];then
    sed -i 's/allow_soft_placement=False/allow_soft_placement=True/g' $script
fi

#net_params
net_params="16,16,32,16 32,16,32,16, 64,16,32,16 128,16,32,16 "
net_params+="16,32,32,16 16,64,32,16 16,128,32,16 "
net_params+="16,16,64,16 16,16,128,16 "
net_params+="16,16,32,32 16,16,32,64 16,16,32,128 "

#list of metrics
metrics="sm__cycles_elapsed.avg.per_second,\
sm__cycles_elapsed.avg,\
sm__inst_executed_pipe_tensor.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
dram__bytes.sum,"

### L1 transactions
# local
metrics+="\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,"
# shared
metrics+="\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,"
# global
metrics+="l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,"
# atomic
metrics+="\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum,"

### L2 transactions
# read + write
metrics+="\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,"
#atomic
metrics+="\
lts__t_sectors_op_atom.sum,\
lts__t_sectors_op_red.sum"

#export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
#export XLA_FLAGS="--xla_dump_to=$run_dir"

cells="lstm"
precs="16 32"

num_warmup=5
num_iter=1

for prec in $precs; do
    for cell in $cells; do
        for net_param in ${net_params}; do
            tmp_param=(${net_param//,/ })
            batch=${tmp_param[0]}
            time=${tmp_param[1]}
            feature=${tmp_param[2]}
            h_size=${tmp_param[3]}
            input_tensor_shape=$batch" "$time" "$feature
            echo $input_tensor_shape

            outputstr=tf1.fp_${prec}.celltype_${cell}.input_${batch}x${time}x${feature}.nneu_${h_size}
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

#!/bin/bash
#SBATCH -J rnn1d-pt
#SBATCH -C gpu
#SBATCH --gres=gpu:1
##SBATCH --exclusive
#SBATCH -t 04:00:00

#activate env
conda activate /global/cfs/cdirs/m1759/charlene/condaenvs/py3.7pt1.5cuda10.2.89
export PROFILER='cupy'

#rankspernode
rankspernode=1

#openmp  
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
sruncmd="srun -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) --cpu_bind=cores"

#create run dir
run_dir=$SCRATCH/tf_cnn_kernels_nsight/Ker-rnn1d-pt-$SLURM_JOBID/
mkdir -p ${run_dir}

#copy relevant files
script_dir=/where BasicKernels is/
script="rnn1d_pt.py"
cp ${script_dir}/python/$script ${run_dir}/
cp $0 ${run_dir}/rnn1d-pt.sh

#step in
cd ${run_dir}

#net_params
net_params="64x64x64,64,16,3 "
#net_params="64x64x1,64,16,3 128x64x1,64,16,3 256x64x1,64,16,3 \
#64x128x1,128,16,3 128x256x1,256,16,3 256x64x1,64,4,3 \
#64x64x1,64,16,5 64x64x1,64,16,7 256x256x1,256,16,7 "
#net_params="64x16x64,32 128x16x64,32 256x16x64,32 "
#net_params+="64x64x64,32 64x16x128,32 64x16x256,32 "
#net_params+="64x16x64,64 64x16x64,128 64x16x64,256 "

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

cells="RNN LSTM GRU "
precs="16 32"
#cells="RNN "
#precs="16 "

num_warmup=5
num_iter=1

for prec in $precs; do
    for cell in $cells; do
    
            #iterate over input tuples
            for net_param in ${net_params}; do 
                tmp_param=(${net_param//,/ })
                input_tensor_shape=${tmp_param[0]//x/ }                 
                input_size=${tmp_param[1]}
                hidden_size=${tmp_param[2]}
                nneu=${tmp_param[3]}

                #iterate over FW BW
                for ctype in calibrate forward backward; do
                    
                    outputstr=pt.fp_${prec}.celltype_${cell}.input_${tmp_param[0]}.nneu_${nneu}.pass_${ctype}
                    
                    #profile string
                    profilestring="/usr/common/software/cuda/11.0.167/bin/nv-nsight-cu-cli \
                    --profile-from-start off --metrics ${metrics} --csv --kernel-base demangled"

                    ${sruncmd} ${profilestring} $(which python) -u ./$script \
                        --input_tensor_shape $input_tensor_shape \
                        --cell_type $cell \
                        --input_size $input_size \
                        --hidden_size $hidden_size \
                        --num_layers $nneu \
                        --dtype float${prec} \
                        --num_iterations $num_iter \
                        --num_warmups $num_warmup \
                        --compute_type ${ctype} 2>&1 > out.${outputstr}
                done
            done
    done
done

#!/bin/bash
#SBATCH -C gpu
#SBATCH -J conv2d-pt
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -q special
#SBATCH -A m1759
#SBATCH -t 2:00:00

#activate env
module load pytorch/v1.5.0-gpu
export PROFILER='cupy'

#rankspernode
rankspernode=1

#openmp  
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
sruncmd="srun -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) --cpu_bind=cores"

run_dir=$PWD/pt_cnn_kernels_nsight/conv2d-pt/$SLURM_JOBID/
mkdir -p ${run_dir}

#copy relevant files
script_dir=../python
script="conv2d_pt.py"
cp $script_dir/$script $run_dir/

#step in
cd ${run_dir}

#net_params
net_params="ResNet50-2,112x112x64,3x3x64x64,1,16 ResNet50-2,112x112x64,3x3x64x64,2,16 ResNet50-2,112x112x64,3x3x64x64,3,16 "
net_params+="ResNet50-2,112x112x64,3x3x64x64,2,32 ResNet50-2,112x112x64,3x3x64x64,2,64 "
net_params+="ResNet50-2,112x112x64,7x7x64x64,2,16 ResNet50-2,112x112x64,9x9x64x64,2,16 "
net_params+="ResNet50-2,112x112x64,3x3x64x128,2,16 ResNet50-2,112x112x64,3x3x64x256,2,16 ResNet50-2,112x112x64,3x3x64x512,2,16 "

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
#data_formats="NHWC NCHW "
precs="16 32"
data_formats="NHWC "

num_warmup=5
num_iter=1

for prec in $precs; do
    for data_format in $data_formats; do

        #iterate over input tuples
        for net_param in ${net_params}; do 
            tmp_param=(${net_param//,/ })
            name=${tmp_param[0]}
            input_tensor_shape=${tmp_param[1]//x/ }
            kernel_shape=${tmp_param[2]//x/ }
            stride=${tmp_param[3]}
            batch_size=${tmp_param[4]}
            
            outputstr1=pt.name_${name}.batch_${tmp_param[4]}.input_${tmp_param[1]}
            outputstr2=kernel_${tmp_param[2]}.stride_${stride}.data_${data_format}.fp${prec}
            
            #iterate over FW BW
            for ctype in calibrate forward backward; do
                #profile string
                profilestring="/usr/common/software/cuda/11.0.167/bin/nv-nsight-cu-cli \
                --profile-from-start off --metrics ${metrics} --kernel-base demangled -o \
                ${outputstr1}.${outputstr2}.pass_${ctype}.${enable_xla}"
                
                ${sruncmd} ${profilestring} $(which python) -u ./$script \
                    --dtype float${prec} \
                    --data_format ${data_format} \
                    --input_tensor_shape ${batch_size} ${input_tensor_shape} \
                    --kernel_shape ${kernel_shape} \
                    --stride ${stride} \
                    --num_warmups ${num_warmup} \
                    --num_iterations ${num_iter} \
                    --compute_type ${ctype}
            done
        done
    done
done


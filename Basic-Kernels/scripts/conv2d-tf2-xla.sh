#!/bin/bash
#SBATCH -J conv2d-tf2-xla
#SBATCH -C gpu
#SBATCH --gres=gpu:1
##SBATCH --exclusive
#SBATCH -t 04:00:00

#activate env
module load tensorflow/gpu-2.2.0-py37
export PROFILER='cupy'

#enalbe XLA or not, 'xla' or 'noxla'
export enable_xla='xla'

#export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
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
run_dir=$SCRATCH/tf_cnn_kernels_nsight/Ker-conv2d-tf2-xla-$SLURM_JOBID/
mkdir -p ${run_dir}

#copy relevant files
script_dir=<where BasicKernels is>
script="conv2d_tf2.py"
cp ${script_dir}/python/$script ${run_dir}/
cp $0 ${run_dir}/conv2d-tf2-$enable_xla.sh

#step in
cd ${run_dir}

#net_params
net_params="ResNet50-2,112x112x64,3x3x64x64,2 ResNet50-2,112x112x64,3x3x64x256,2 ResNet50-2,112x112x64,7x7x64x64,2 "
#net_params="VGG-1,224x224x3,3x3x3x64,1 VGG-2,224x224x64,3x3x64x128,2 VGG-3,112x112x128,3x3x128x256,2 ResNet50-1,224x224x3,7x7x3x64,2 ResNet50-2,112x112x64,3x3x64x64,2 "
#net_params+="ResNet50-2,112x112x64,3x3x64x128,2 ResNet50-2,112x112x64,3x3x64x256,2 ResNet50-2,112x112x64,3x3x64x512,2 ResNet50-2,112x112x64,7x7x64x64,2 ResNet50-2,112x112x64,9x9x64x64,2 "
#net_params+="ResNet50-2,112x112x64,3x3x64x128,1 ResNet50-2,112x112x64,3x3x64x256,1 ResNet50-2,112x112x64,3x3x64x512,1 ResNet50-2,112x112x64,7x7x64x64,1 ResNet50-2,112x112x64,9x9x64x64,1 "
#net_params+="ResNet50-2,112x112x64,3x3x64x128,3 ResNet50-2,112x112x64,3x3x64x256,3 ResNet50-2,112x112x64,3x3x64x512,3 ResNet50-2,112x112x64,7x7x64x64,3 ResNet50-2,112x112x64,9x9x64x64,3 "

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
#precs="16 32 "
#batch_sizes="64 128 256 "
#data_formats="NHWC NCHW "
precs="16 "
batch_sizes="64 "
data_formats="NHWC "

num_warmup=5
num_iter=1

for prec in $precs; do
    for batch_size in $batch_sizes; do
        for data_format in $data_formats; do
    
            #iterate over input tuples
            for net_param in ${net_params}; do 
                tmp_param=(${net_param//,/ })
                name=${tmp_param[0]}
                input_tensor_shape=${tmp_param[1]//x/ }
                kernel_shape=${tmp_param[2]//x/ }
                stride=${tmp_param[3]}

                outputstr1=tf2.name_${name}.batch_${batch_size}.input_${tmp_param[1]}
                outputstr2=kernel_${tmp_param[2]}.stride_${stride}.data_${data_format}.fp${prec}

                #iterate over FW BW
                for ctype in calibrate forward backward; do

                    #profile string
                    profilestring="/usr/common/software/cuda/11.0.167/bin/nv-nsight-cu-cli \
                    --profile-from-start off --metrics ${metrics} --csv --kernel-base demangled"

                    if [ $enable_xla == "xla" ];then
                        ${sruncmd} ${profilestring} $(which python) -u ./$script \
                            --dtype float${prec} \
                            --data_format ${data_format} \
                            --input_tensor_shape ${batch_size} ${input_tensor_shape} \
                            --kernel_shape ${kernel_shape} \
                            --stride ${stride} \
                            --num_warmups ${num_warmup} \
                            --num_iterations ${num_iter} \
                            --enable_xla \
                            --compute_type ${ctype} 2>&1 > out.${outputstr1}.${outputstr2}.pass_${ctype}.${enable_xla}
                    else
                        ${sruncmd} ${profilestring} $(which python) -u ./$script \
                            --dtype float${prec} \
                            --data_format ${data_format} \
                            --input_tensor_shape ${batch_size} ${input_tensor_shape} \
                            --kernel_shape ${kernel_shape} \
                            --stride ${stride} \
                            --num_warmups ${num_warmup} \
                            --num_iterations ${num_iter} \
                            --compute_type ${ctype} 2>&1 > out.${outputstr1}.${outputstr2}.pass_${ctype}.${enable_xla}
                    fi

                done
            done

        done
    done
done

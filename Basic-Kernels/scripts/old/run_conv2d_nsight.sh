#!/bin/bash
#SBATCH -J conv2d_test
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive

#load modules
module load cuda/10.2.89
module load python/3.7-anaconda-2019.07

#activate env
source activate py3.7-tf2-cuda-10.2.89

#rankspernode
rankspernode=1

#custome link
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/../lib

#openmp stuff
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
sruncmd="srun -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 80 / ${rankspernode} )) --cpu_bind=cores"


#create run dir
run_dir=$PWD/tf_cnn_kernels_nsight/runs/${SLURM_JOBID}
mkdir -p ${run_dir}

#copy relevant files
cp ../python/conv2d_v2.py ${run_dir}/

#variables
prec=16
batch_size=64
data_format="NHWC"

#net_params
#net_params="VGG-1,224x224x3,3x3x3x64,1 ResNet50-1,224x224x3,7x7x3x64,2 VGG-2,224x224x64,3x3x64x128,2 VGG-3,112x112x128,3x3x128x256,2 ResNet50-2,112x112x64,3x3x64x64,2"
#net_params="ResNet50-2,112x112x64,3x3x64x128,2 ResNet50-2,112x112x64,3x3x64x256,2 ResNet50-2,112x112x64,3x3x64x512,2 ResNet50-2,112x112x64,7x7x64x64,2 ResNet50-2,112x112x64,9x9x64x64,2"
#net_params="ResNet50-2,112x112x64,3x3x64x128,1 ResNet50-2,112x112x64,3x3x64x256,1 ResNet50-2,112x112x64,3x3x64x512,1 ResNet50-2,112x112x64,7x7x64x64,1 ResNet50-2,112x112x64,9x9x64x64,1"
#net_params="ResNet50-2,112x112x64,3x3x64x128,3 ResNet50-2,112x112x64,3x3x64x256,3 ResNet50-2,112x112x64,3x3x64x512,3 ResNet50-2,112x112x64,7x7x64x64,3 ResNet50-2,112x112x64,9x9x64x64,3"
#net_params="ResNet50-2,112x112x64,3x3x64x64,1 ResNet50-2,112x112x64,3x3x64x64,3"
net_params="ResNet50-2,112x112x64,3x3x64x64,2"

#step in
cd ${run_dir}

#list of metrics
metrics="sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
smsp__cycles_elapsed.sum,\
smsp__cycles_elapsed.sum.per_second,\
smsp__pipe_tensor_op_hmma_cycles_active.sum,\
smsp__pipe_tensor_op_hmma_cycles_active.sum.per_second,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum,\
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom.sum,\
l1tex__t_set_accesses_pipe_tex_mem_surface_op_red.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sectors_op_atom.sum,\
lts__t_sectors_op_red.sum,\
dram__sectors_read.sum,\
dram__sectors_write.sum
"

#iterate over metrics
for metric in ${metrics}; do
    
    #iterate over input tuples
    for input in ${net_params}; do 
        OLDIFS=$IFS; IFS=','
        set -- $input; 
        name=$1
        input_tensor_shape=${2//x/ }
        kernel_shape=${3//x/ }
        stride=${4}
        IFS=${OLDIFS}

        #iterate over FW BW
        for ctype in calibrate forward backward; do

            #get better metric name
            metricname=${metric//,/-}

            profilestring="nv-nsight-cu-cli --profile-from-start off --metrics ${metric} -f"
            metrictag="metrics"

            #profilestring=${profilestring}" -f -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}.metric_${metricname}"
            profilestring=${profilestring}" -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}"

            #compute types
            ${sruncmd} ${profilestring} $(which python) -u ./conv2d_v2.py \
                --dtype float${prec} \
                --data_format ${data_format} \
                --input_tensor_shape ${batch_size} ${input_tensor_shape} \
                --kernel_shape ${kernel_shape} \
                --stride ${stride} \
                --num_warmups 5 \
                --num_iterations 20 \
                --compute_type ${ctype}

        done

    done

done

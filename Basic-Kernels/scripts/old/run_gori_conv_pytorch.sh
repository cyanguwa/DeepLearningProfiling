#!/bin/bash
#SBATCH -J conv2d_test
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive

#load modules
#module unload cuda
module load cuda/10.1.243
#module load cuda/10.1.168
module load python/3.7-anaconda-2019.07
module load pytorch/v1.3.1-gpu

#activate env
source activate py3.7-tf2
#module load tensorflow/gpu-1.13.1-py36
#module load tensorflow/gpu-2.0.0-beta-py36

#rankspernode
rankspernode=1

#openmp stuff
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
sruncmd="srun -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 80 / ${rankspernode} )) --cpu_bind=cores"


#create run dir
run_dir=$PWD/tf_cnn_kernels_2/runs/${SLURM_JOBID}
mkdir -p ${run_dir}

#copy relevant files
cp ../python/conv2d_pytorch.py ${run_dir}/

#variables
prec=16
batch_size=16
data_format="NHWC"

#net_params
net_params="VGG-1,224x224x3,3x3x3x64,1 ResNet50-1,224x224x3,7x7x3x64,2 VGG-2,224x224x64,3x3x64x128,2 VGG-3,112x112x128,3x3x128x256,2 ResNet50-2,112x112x64,3x3x64x64,2"
#net_params_2="ResNet50-2,112x112x64,3x3x64x128,2 ResNet50-2,112x112x64,3x3x64x256,2 ResNet50-2,112x112x64,3x3x64x512,2 ResNet50-2,112x112x64,7x7x64x64,2 ResNet50-2,112x112x64,9x9x64x64,2"
#net_params_3="ResNet50-2,112x112x64,3x3x64x128,1 ResNet50-2,112x112x64,3x3x64x256,1 ResNet50-2,112x112x64,3x3x64x512,1 ResNet50-2,112x112x64,7x7x64x64,1 ResNet50-2,112x112x64,9x9x64x64,1"
#net_params_4="ResNet50-2,112x112x64,3x3x64x128,3 ResNet50-2,112x112x64,3x3x64x256,3 ResNet50-2,112x112x64,3x3x64x512,3 ResNet50-2,112x112x64,7x7x64x64,3 ResNet50-2,112x112x64,9x9x64x64,3"
#net_params="ResNet50-2,112x112x64,3x3x64x64,2 ResNet50-2,112x112x64,3x3x64x64,3"
#net_params="ResNet50-2,112x112x64,3x3x64x128,2"


#step in
cd ${run_dir}

#list of metrics
#metrics=""
metrics=\
"
time \
sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,\
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
dram__sectors_read.sum,\
dram__sectors_write.sum,\
lts__t_sectors_aperture_sysmem_op_read.sum,\
lts__t_sectors_aperture_sysmem_op_write.sum\
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

            #assemble profiling string
            if [ "${metric}" == "time" ]; then
                profilestring="nsys profile --trace=cublas,cuda,cudnn,osrt --capture-range=cudaProfilerApi --stats=true -f true"
                metrictag="time"
                #profilestring="nv-nsight-cu-cli"
            else
                profilestring="nv-nsight-cu-cli --profile-from-start off --metrics ${metric} -f"
                metrictag="metrics"
                #profilestring="nv-nsight-cu-cli --metrics ${metric}"
            fi
            #profilestring=${profilestring}" -f -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}.metric_${metricname}"
            profilestring=${profilestring}" -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}.metric_${metrictag}"

            #compute types
            ${sruncmd} ${profilestring} $(which python) -u ./conv2d_pytorch.py \
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

#copy results
#cp *.nvvp /project/projectdirs/mpccc/tkurth/Profiling/tf_perf_kernels/data/good_new/

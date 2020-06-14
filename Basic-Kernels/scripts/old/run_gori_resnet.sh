#!/bin/bash
#SBATCH -J conv2d_test
#SBATCH -t 02:00:00
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive

#load modules
module unload cuda
module load cuda/10.0.130
module load python/3.6-anaconda-4.4

#activate env
source activate thorstendl-gori-py3-tf
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
run_dir=$WORK/tf_cnn_kernels_2/runs/${SLURM_JOBID}
mkdir -p ${run_dir}

#copy relevant files
cp resnet.py ${run_dir}/

#variables
prec=16
batch_size=16
data_format="NHWC"

#set growth
#export TF_FORCE_GPU_ALLOW_GROWTH=true

#net_params
net_params="ResNet50,224x224x3,100,16"
#net_params="VGG-1,224x224x3,3x3x3x64,1 ResNet50-1,224x224x3,7x7x3x64,2 VGG-2,224x224x64,3x3x64x128,2 VGG-3,112x112x128,3x3x128x256,2 ResNet50-2,112x112x64,3x3x64x64,2"
#net_params="ResNet50-2,112x112x64,7x7x64x64,2"
#net_params="ResNet50-2,112x112x64,3x3x64x128,2 ResNet50-2,112x112x64,3x3x64x256,2 ResNet50-2,112x112x64,3x3x64x512,2 ResNet50-2,112x112x64,7x7x64x64,2 ResNet50-2,112x112x64,9x9x64x64,2"
#net_params="ResNet50-2,112x112x64,3x3x64x128,1 ResNet50-2,112x112x64,3x3x64x256,1 ResNet50-2,112x112x64,3x3x64x512,1 ResNet50-2,112x112x64,7x7x64x64,1 ResNet50-2,112x112x64,9x9x64x64,1"
#net_params_4="ResNet50-2,112x112x64,3x3x64x128,3 ResNet50-2,112x112x64,3x3x64x256,3 ResNet50-2,112x112x64,3x3x64x512,3 ResNet50-2,112x112x64,7x7x64x64,3 ResNet50-2,112x112x64,9x9x64x64,3"
#net_params="ResNet50-2,112x112x64,3x3x64x64,2" #ResNet50-2,112x112x64,3x3x64x64,3"
#net_params="ResNet50-2,112x112x64,3x3x64x128,2 ResNet50-2,112x112x64,3x3x64x256,2 ResNet50-2,112x112x64,3x3x64x512,2"
#net_params="ResNet50-2,112x112x64,9x9x64x64,1" #ResNet50-2,112x112x64,9x9x64x64,2 ResNet50-2,112x112x64,9x9x64x64,3"
#net_params="ResNet50-2,112x112x64,9x9x64x64,2 ResNet50-2,112x112x64,9x9x64x64,3"
#net_params="ResNet50-2,112x112x64,3x3x64x64,2 ResNet50-2,112x112x64,7x7x64x64,2 ResNet50-2,112x112x64,7x7x64x64,3"


#step in
cd ${run_dir}

#list of metrics
metrics="time"
#"tensor_precision_fu_utilization flop_count_hp flop_count_sp sysmem_read_transactions sysmem_write_transactions dram_read_transactions dram_write_transactions l2_read_transactions l2_write_transactions gld_transactions gst_transactions shared_load_transactions shared_store_transactions atomic_transactions"
#metrics="l2_write_transactions"
#metrics="atomic_transactions"
#metrics="l2_tex_read_transactions l2_tex_write_transactions l2_read_transactions l2_write_transactions gld_transactions gst_transactions"
#metrics="shared_load_transactions shared_store_transactions"
#"local_store_transactions local_load_transactions"
#metrics="sm__pipe_tensor_cycles_active.sum" #"smsp__sass_thread_inst_executed_op_fadd_pred_on.sum" #,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"

#iterate over metrics
for metric in ${metrics}; do
    
    #iterate over input tuples
    for input in ${net_params}; do 
        OLDIFS=$IFS; IFS=','
        set -- $input; 
        name=$1
        input_tensor_shape=${2//x/ }
        num_classes=${3//x/ }
        stride=${4}
        IFS=${OLDIFS}

        #iterate over FW BW
        for ctype in forward calibrate forward backward; do

            #get better metric name
            metricname=${metric//,/-}
    
            #assemble profiling string
            if [ "${metric}" == "time" ]; then
                profilestring="nvprof --profile-from-start off"
            else
                profilestring="nvprof --profile-from-start off --metrics ${metric}"
            fi
            profilestring=${profilestring}" -f -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}.metric_${metricname}.nvvp"
            #profilestring=""

            #compute types
            ${sruncmd} ${profilestring} $(which python) -u ./resnet.py \
                --dtype float${prec} \
                --data_format ${data_format} \
                --input_tensor_shape ${batch_size} ${input_tensor_shape} \
                --num_classes ${num_classes} \
                --num_warmups 5 \
                --num_iterations 20 \
                --compute_type ${ctype}
            break
        done
    done
done

#copy results
#cp *.nvvp /project/projectdirs/mpccc/tkurth/Profiling/tf_perf_kernels/data/good_new/

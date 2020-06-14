#!/bin/bash
#SBATCH -J conv2d-TF
#SBATCH -t 02:00:00
#SBATCH -C gpu
#SBATCH --gres=gpu:1
##SBATCH --exclusive

#load modules
#module load cuda/10.2.89
#module load cuda/11.0.167
#module load python/3.7-anaconda-2019.07

# TensorFlow version 1 or 2
export TFversion=2

#activate env
#source activate py3.7-tf2-cuda-10.2.89
if [ $TFversion == "1" ] 
then	
	module load tensorflow/gpu-1.15.0-py37
	export PROFILER='cupy'
elif [ $TFversion == "2" ]
then
	module load tensorflow/gpu-2.2.0-py37
	#pip install --user cupy-cuda101
	export PROFILER='cupy'
fi
#export XLA_FLAGS="--xla_dump_to=/global/cscratch1/sd/cjyang/tf_cnn_kernels_nsight/Ker-conv2d-TF-739230"

#rankspernode
rankspernode=1

#custome link
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/common/software/cuda/10.0.130/lib64 
#${PWD}/../lib

#openmp stuff
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
#sruncmd="srun -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 80 / ${rankspernode} )) --cpu_bind=cores"
sruncmd="srun -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) --cpu_bind=cores"

#create run dir
run_dir=$SCRATCH/tf_cnn_kernels_nsight/Ker-conv2d-TF${TFversion}-$SLURM_JOBID/
mkdir -p ${run_dir}

#copy relevant files
script_dir=/global/cfs/cdirs/nstaff/cjyang/study/Yunsong/tf-perf-kernels/
if [ $TFversion == "1" ] 
then	
	script="conv2d_tf1.py"
elif [ $TFversion == "2" ]
then
	script="conv2d_tf2.py"
fi
cp ${script_dir}/python/$script ${run_dir}/
cp $0 ${run_dir}/Ker-conv2d-TF${TFversion}.sh

#step in
cd ${run_dir}

#variables
prec=16
batch_size=64
data_format="NHWC"

#net_params
net_params="VGG-1,224x224x3,3x3x3x64,1 VGG-2,224x224x64,3x3x64x128,2 VGG-3,112x112x128,3x3x128x256,2 ResNet50-1,224x224x3,7x7x3x64,2 ResNet50-2,112x112x64,3x3x64x64,2 "
net_params+="ResNet50-2,112x112x64,3x3x64x128,2 ResNet50-2,112x112x64,3x3x64x256,2 ResNet50-2,112x112x64,3x3x64x512,2 ResNet50-2,112x112x64,7x7x64x64,2 ResNet50-2,112x112x64,9x9x64x64,2 "
net_params+="ResNet50-2,112x112x64,3x3x64x128,1 ResNet50-2,112x112x64,3x3x64x256,1 ResNet50-2,112x112x64,3x3x64x512,1 ResNet50-2,112x112x64,7x7x64x64,1 ResNet50-2,112x112x64,9x9x64x64,1 "
net_params+="ResNet50-2,112x112x64,3x3x64x128,3 ResNet50-2,112x112x64,3x3x64x256,3 ResNet50-2,112x112x64,3x3x64x512,3 ResNet50-2,112x112x64,7x7x64x64,3 ResNet50-2,112x112x64,9x9x64x64,3 "
#net_params="ResNet50-2,112x112x64,3x3x64x64,1 ResNet50-2,112x112x64,3x3x64x64,3"
#net_params="ResNet50-2,112x112x64,3x3x64x64,2"

#list of metrics
#metrics="sm__cycles_elapsed.avg.per_second \
#sm__cycles_elapsed.avg \
#sm__inst_executed_pipe_tensor.sum.per_cycle_elapsed \
#sm__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed \
#sm__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed \
#sm__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed \
#sm__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed \
#sm__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed \
#sm__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed \
#dram__bytes.sum.per_second \
#lts__t_bytes.sum.per_second \
#l1tex__t_bytes.sum.per_second "
#metrics="sm__cycles_elapsed.avg.per_second \
#sm__cycles_elapsed.avg \
#sm__inst_executed_pipe_tensor.sum \
#sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
#sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
#sm__sass_thread_inst_executed_op_fmul_pred_on.sum \
#sm__sass_thread_inst_executed_op_hadd_pred_on.sum \
#sm__sass_thread_inst_executed_op_hfma_pred_on.sum \
#sm__sass_thread_inst_executed_op_hmul_pred_on.sum \
#dram__bytes.sum \
#lts__t_bytes.sum \
#l1tex__t_bytes.sum "
#metrics="sm__cycles_elapsed.avg.per_second "
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


#iterate over metrics
#for metric in ${metrics}; do
    
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
        #for ctype in calibrate ; do

            #get better metric name
            #metricname=${metric//,/-}

            profilestring="/usr/common/software/cuda/11.0.167/bin/nv-nsight-cu-cli --profile-from-start off --metrics ${metrics} --csv --kernel-base demangled"

            #compute types
	    outputstr1=tf_${TFversion}.name_${name}.batch_${batch_size}.input_${2}
	    outputstr2=kernel_${3}.stride_${4}.data_${data_format}.fp${prec}.pass_${ctype}

            ${sruncmd} ${profilestring} $(which python) -u ./$script \
                --dtype float${prec} \
                --data_format ${data_format} \
                --input_tensor_shape ${batch_size} ${input_tensor_shape} \
                --kernel_shape ${kernel_shape} \
                --stride ${stride} \
                --num_warmups 5 \
                --num_iterations 1 \
                --compute_type ${ctype} 2>&1 > out.${outputstr1}.${outputstr2}.noxla

	    if [ $TFversion == "2" ]
	    then
            ${sruncmd} ${profilestring} $(which python) -u ./$script \
                --dtype float${prec} \
                --data_format ${data_format} \
                --input_tensor_shape ${batch_size} ${input_tensor_shape} \
                --kernel_shape ${kernel_shape} \
                --stride ${stride} \
                --num_warmups 5 \
                --num_iterations 1 \
		--enable_xla \
                --compute_type ${ctype} 2>&1 > out.${outputstr1}.${outputstr2}.xla
    	   fi

        done

    done

#done

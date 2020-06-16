#!/bin/bash
#SBATCH -J O1-LAMB-fw
#SBATCH -C gpu
#SBATCH --gres=gpu:1
##SBATCH --exclusive
#SBATCH -t 04:00:00

# conda activate py37pytorch 
conda activate /global/cfs/cdirs/m1759/charlene/condaenvs/py3.7pt1.5cuda10.2.89

#parameters
run_tag="fw"
data_dir_prefix="/global/cfs/cdirs/nstaff/cjyang/study/Yunsong/mlperf"
output_dir=$SCRATCH/deepcam-benchmark/PT-fp32-O1-LAMB-fw-$SLURM_JOBID

#create files
mkdir -p ${output_dir}

# Prepare the run directory
script_dir=/global/cfs/cdirs/nstaff/cjyang/study/Yunsong/mlperf-deepcam/src/deepCam/
cp -r ${script_dir}/architecture ${output_dir}/
cp -r ${script_dir}/data ${output_dir}/
cp -r ${script_dir}/utils ${output_dir}/
cp ${script_dir}/profile_hdf5_ddp.py ${output_dir}/
cp ${script_dir}/train_hdf5_ddp.py ${output_dir}/ 
cp $0 ${output_dir}/fp32-O1-LAMB-fw.sh
cd ${output_dir}

metrics="sm__cycles_elapsed.avg.per_second \
sm__cycles_elapsed.avg \
sm__inst_executed_pipe_tensor.sum \
sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
sm__sass_thread_inst_executed_op_ffma_pred_on.sum \
sm__sass_thread_inst_executed_op_fmul_pred_on.sum \
sm__sass_thread_inst_executed_op_hadd_pred_on.sum \
sm__sass_thread_inst_executed_op_hfma_pred_on.sum \
sm__sass_thread_inst_executed_op_hmul_pred_on.sum \
dram__bytes.sum \
lts__t_bytes.sum \
l1tex__t_bytes.sum "

echo 'python version' `which python`

for metric in ${metrics}; do

    profilecmd="/project/projectdirs/m1759/nsight-compute-2020.1.0.20/nv-nsight-cu-cli --profile-from-start off --metrics ${metric} --csv --kernel-base demangled"

    srun -n 1 --cpu_bind=cores  -u \
        ${profilecmd} \
        `which python` -u ./profile_hdf5_ddp.py \
        --wireup_method "nccl-slurm-pmi" \
        --run_tag ${run_tag} \
        --data_dir_prefix ${data_dir_prefix} \
        --output_dir ${output_dir} \
        --max_inter_threads 0 \
        --optimizer "LAMB" \
        --start_lr 1e-3 \
        --num_warmup_steps 5 \
        --num_profile_steps 1 \
        --profile "Forward" \
        --weight_decay 1e-2 \
        --amp_opt_level O1 \
        --local_batch_size 2 2>&1 > ${output_dir}/out.${metric} 

done 

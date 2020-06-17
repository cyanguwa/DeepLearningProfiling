# DeepLearningProfiling

1. This repository includes three components:
- /Basic-Kernels: conv2d and rnn1d (RNN, LSTM, GRU) kernels from TensorFlow 1.15, TensorFlow 2.2 and PyTorch 1.5  
  [The code is based on: https://github.com/NERSC/tf-perf-kernels]
- /climate-seg-benchmark: climate image segmentation code, implemented using TensorFlow 1.15  
  [The code is from: https://github.com/sparticlesteve/climate-seg-benchmark]
- /mlperf-deepcam: climate image segmentation code, implemented using PyTorch  
  [The code is from: https://github.com/azrael417/mlperf-deepcam]

2. Scripts for profiling, post-processing and Roofline plotting are added on top of the original repositories. Some of the profiling scripts are based on:
- https://gitlab.com/NERSC/roofline-on-nvidia-gpus/

3. The new hierarchical Roofline methodology is: [Based on Nsight Compute from CUDA 11]
- Time: sm__cycles_elapsed.avg / sm__cycles_elapsed.avg.per_second
- FLOPs:  
  sm__sass_thread_inst_executed_op_dadd_pred_on.sum + 2 x sm__sass_thread_inst_executed_op_dfma_pred_on.sum + sm__sass_thread_inst_executed_op_dmul_pred_on.sum + sm__sass_thread_inst_executed_op_fadd_pred_on.sum + 2 x sm__sass_thread_inst_executed_op_ffma_pred_on.sum + sm__sass_thread_inst_executed_op_fmul_pred_on.sum + sm__sass_thread_inst_executed_op_hadd_pred_on.sum + 2 x sm__sass_thread_inst_executed_op_hfma_pred_on.sum + sm__sass_thread_inst_executed_op_hmul_pred_on.sum + 512 x sm__inst_executed_pipe_tensor.sum
- Bytes: dram__bytes.sum, lts__t_bytes.sum, and l1tex__t_bytes.sum

4. Some notes on the file structure:
- TF-xxx and PT-xxx contain the results for all 12 metrics for different configurations
- code-xxx contains the source code, same as the original repository or modified
- scripts-xxx contains the job scripts run on Cori, for different configurations
- genJobScripts-xx.ipynb generates Slurm job scripts for scripts-xxx folder
- plotRoofline.py	contains the roofline() function for plotting
- postprocess-xx.ipynb processes the results in TF-xxx and PT-xxx folders, and put them in relevant Pandas dataframes, dfcsfw and dfcsbw for example.

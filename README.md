# DeepLearningProfiling
This repository includes three components:
- /Basic-Kernels: conv2d and rnn1d (RNN, LSTM, GRU) kernels from TensorFlow 1.15, TensorFlow 2.2 and PyTorch 1.5

  [The code is based on: https://github.com/NERSC/tf-perf-kernels]
- /climate-seg-benchmark: climate image segmentation code, implemented using TensorFlow 1.15
  
  [The code is from: https://github.com/sparticlesteve/climate-seg-benchmark]
- /mlperf-deepcam: climate image segmentation code, implemented using PyTorch
  
  [The code is from: https://github.com/azrael417/mlperf-deepcam]


Scripts for profiling, post-processing and Roofline plotting are added on top of the original repositories. Some of the profiling scripts are based on:
- https://gitlab.com/NERSC/roofline-on-nvidia-gpus/

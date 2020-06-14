#!/bin/bash

#GPU
#clean up
docker rm -f $(docker ps -aq)

#build
docker build -t registry.services.nersc.gov/tkurth/tf_perf_kernels:gpu-latest -f ./Dockerfile .

#push
docker push registry.services.nersc.gov/tkurth/tf_perf_kernels:gpu-latest

#login
#docker run -it --name test registry.services.nersc.gov/tkurth/exalearn_rl:latest /bin/bash

#CPU:
#build
#docker build -t registry.services.nersc.gov/tkurth/exalearn_rl:cpu-latest -f ./Dockerfile_cpu .

#push
#docker push registry.services.nersc.gov/tkurth/exalearn_rl:cpu-latest

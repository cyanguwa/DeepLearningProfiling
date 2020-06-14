## a conv2d kernel using tensorflow
# https://pytorch.org/xla/release/1.5/index.html
# xla_model

# torch_xla.core.xla_model.xla_device(n=None, devkind=None)[SOURCE]
# Returns a given instance of an XLA device.

# Parameters
# n (python:int, optional) – The specific instance (ordinal) to be returned. If specified, the specific XLA device instance will be returned. Otherwise the first device of devkind will be returned.
# devkind (string..., optional) – If specified, one of TPU, GPU or CPU (the ‘GPU’ XLA device is currently not implemented).
# Returns
# A torch.device with the requested instance.

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# import torch_xla
# import torch_xla.core.xla_model as xm

#from tensorflow.python.eager import profiler as tfprof
import numpy as np
import argparse
import time
if os.environ['PROFILER'] == 'pycuda':
    try:
        import pycuda.autoinit
        import pycuda as pyc
        have_pycuda=True
        print("pycuda enabled")
    except:
        print("pycuda not installed")
        have_pycuda=False
elif os.environ['PROFILER'] == 'cupy':
    try:
        import cupy 
        have_cupy=True
        print("cupy enabled")
    except:
        print("cupy not installed")
        have_cupy=False
else:
    print('Please make sure Start/Stop profiler is installed')

# print('xla device:')
# print(xm.xla_device())
#warnings.simplefilter('ignore')


#def conv2d(input_data, data_format, weights, stride_, dtype):
#    if data_format == "NCHW":
#        input_data = tf.transpose(input_data, [0,3,1,2])
#        strides = [1,1,stride_, stride_]
#    else:
#        strides = [1,stride_,stride_,1]
#    output_data = tf.nn.conv2d(input_data, weights, strides=strides, padding='SAME', data_format=data_format)
#    return output_data


class CNN(nn.Module):
    def __init__(self, kernel_shape, stride):
        super(CNN, self).__init__()
        self.input_c = kernel_shape[2]
        self.output_c = kernel_shape[3]
        self.kernel_size = kernel_shape[0]
        self.stride = stride
        self.conv2d = nn.Conv2d(self.input_c, self.output_c, kernel_size=self.kernel_size, stride=self.stride)
        
    def forward(self,x):
        x = self.conv2d(x)
        return x

#conv2d = CNN(kernel_shape, stride)

    

#calibration measurement
def run_calibrate(input_tensor_shape, data_format, weights, stride, dtype, kernel_shape, device):
    #define op
    #run the stuff

    input_image = np.random.randn(input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2])
    input_image = torch.FloatTensor(input_image)

    input_image = input_image.unsqueeze(0)

    if device[0:4] == 'cuda':
        input_image2 = input_image.to(device)
        _ = input_image.cpu().detach().numpy()
    else:
    #  input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=dtype)
        _ = input_image.detach().numpy()


#forward
def run_forward(input_tensor_shape, data_format, weights, stride, dtype, kernel_shape, device):
    #define op
    #input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=dtype)

    input_image = np.random.randn(input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2])
    input_image = torch.FloatTensor(input_image)

    input_image = input_image.unsqueeze(0)
    conv2d = CNN(kernel_shape, stride)

    if device[0:4] == 'cuda':
        input_image = input_image.to(device)
        conv2d.to(device)
        output_result = conv2d(input_image)
        _ = output_result.cpu().detach().numpy()
    else:
        output_result = conv2d(input_image)
        _ = output_result.detach().numpy()


#backward
def run_backward(input_tensor_shape, data_format, weights, stride, dtype, kernel_shape, device):
    #define op, under tape
    #input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=dtype)

    input_image = np.random.randn(input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2])
    input_image = torch.FloatTensor(input_image)

    input_image = input_image.unsqueeze(0)
    conv2d = CNN(kernel_shape, stride)

    lr = 0.01
    momentum = 0.5
    optimizer = optim.SGD(conv2d.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()


    if device[0:4] == 'cuda':
        input_image = input_image.to(device)
        conv2d.to(device)
        output_result = conv2d(input_image)
        res = torch.sum(output_result)
        res.backward()
        optimizer.step()
        _, _ = res.cpu().detach().numpy(), input_image.cpu().detach().numpy()
    else:
        output_result = conv2d(input_image)
        res = torch.sum(output_result)
        res.backward()
        optimizer.step()
        _, _ = res.detach().numpy(), input_image.detach().numpy()

    #grad_input   = tape.gradient(output_result, input_image)
    #grad_weights = tape.gradient(output_result, weights)
    
    #run the stuff


def main(input_tensor_shape, data_format, kernel_shape, stride, dtype, n_iter, n_warm, compute_type): #, enable_xla):
    
    #datatype selection
    if dtype == 'float16':
        tensor_type=torch.float16
    elif dtype == 'float32':
        tensor_type=torch.float32
    else:
        raise Exception('data type can only be float16 or float32')
    
    ##XLA or not
    if torch.cuda.device_count():
        device = 'cuda:0' #if enable_xla else 'cuda:0'
    else:
        device = 'cpu:0' #if enable_xla else 'cpu:0'
        
    print("Running on device {}".format(device))
    #tf.config.experimental.set_memory_growth(device, True)
    #tf.config.gpu.set_per_process_memory_growth(True)

    # select commpute type
    if compute_type == "forward":
        compfunc = run_forward
    elif compute_type == "backward":
        compfunc = run_backward
    elif compute_type == "calibrate":
        compfunc = run_calibrate
    else:
        raise ValueError("Error, compute_type should be either forward or backward or calibrate")


    print(device)
    #we might need that
    #with torch.device('cuda:0'):
    print("warming up for {} steps".format(n_warm))
    weights = np.random.randn(kernel_shape[0],kernel_shape[1],kernel_shape[2],kernel_shape[3])
    weights = torch.FloatTensor(weights)
    
    #start session
    #with torch.device(device):
    start = time.time()
    for i in range(n_warm):
        compfunc(input_tensor_shape, data_format, weights, stride, tensor_type, kernel_shape, device)
    end = time.time()
    print("done")
    duration = end-start
    print('Warmup {:.2f} seconds, {:.2f} seconds/iter'.format(duration, duration/float(n_warm)))
    
    print("running for {} steps".format(n_iter))
    start = time.time()
    #start profiling
    if os.environ['PROFILER'] == 'pycuda':
        if have_pycuda:
            pyc.driver.start_profiler()
    elif os.environ['PROFILER'] == 'cupy':
        if have_cupy:
            cupy.cuda.profiler.start()
    #with torch.device(device):
    for i in range(n_iter):
        compfunc(input_tensor_shape, data_format, weights, stride, tensor_type, kernel_shape, device)

    #stop profiling
    if os.environ['PROFILER'] == 'pycuda':
        if have_pycuda:
            pyc.driver.stop_profiler()
    elif os.environ['PROFILER'] == 'cupy':
        if have_cupy:
            cupy.cuda.profiler.stop()
    end = time.time()
    print("done")
    
    duration = end-start
    print('Run {:.2f} seconds, {:.2f} seconds/iter'.format(duration, duration/float(n_iter)))



if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', default=[1, 256, 256], help='the shape of the input tensor. Note that it depends on data_format (default NHWC)')
    AP.add_argument('--data_format', type=str, default='NHWC', help='choose either channels_last or channels_first')
    AP.add_argument('--kernel_shape', type=int, nargs='+', default=[5,5,1,32], help='the shape of the conv kernel [filter_height, filter_width, in_channels, out_channels]')
    AP.add_argument('--stride', type=int, default=1, help='the stride')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="forward", help='forward or backward pass')
#     AP.add_argument('--enable_xla', action="store_true", help="enable XLA support")
    parsed = AP.parse_args()
    
    #print args
    for arg in vars(parsed):
        print(arg, ":", getattr(parsed, arg))
        
    
    main(input_tensor_shape=parsed.input_tensor_shape,
         data_format=parsed.data_format,
         kernel_shape=parsed.kernel_shape,
         stride=parsed.stride,
         dtype=parsed.dtype,
         n_iter=parsed.num_iterations,
         n_warm=parsed.num_warmups,
         compute_type=parsed.compute_type)
#     ,
#          enable_xla=parsed.enable_xla)
    
    


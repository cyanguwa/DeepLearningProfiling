# A rnn1d kernel using pytorch

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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

# warnings.simplefilter('ignore')

def run_calibrate(input_image, myRNN):
    output_result = input_image

def run_forward(input_image, myRNN):
    output_result = myRNN(input_image)
    
def run_backward(input_image, myRNN):
    lr = 0.01
    momentum = 0.5
    optimizer = optim.SGD(myRNN.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()
        
    output_result, _ = myRNN(input_image)
    output_result.sum().backward()
    optimizer.step()
    
def main(input_tensor_shape, cell_type, n_neurons, dtype, n_iter, n_warm, compute_type): 

    if dtype == 'float16':
        tensor_type=torch.float16
    elif dtype == 'float32':
        tensor_type=torch.float32
    else:
        raise Exception('data type can only be float16 or float32')
    
    if torch.cuda.device_count():
        device = torch.device('cuda:0') 
    else:
        device = torch.device('cpu')
    print("Running on device {}".format(device))
        
    # the input format is (batch,time_steps,features)
    input_image = torch.randn(input_tensor_shape[0],input_tensor_shape[1],input_tensor_shape[2],device=device,dtype=tensor_type)
    input_image = input_image.float().cuda()

    input_size = input_tensor_shape[2]
    hidden_size = n_neurons

    # init rnn kernel
    if cell_type == 'lstm':
        myRNN = nn.LSTM(input_size, hidden_size, batch_first=True)
    elif cell_type == 'rnn':
        myRNN = nn.RNN(input_size, hidden_size, batch_first=True)
    elif cell_type == 'gru':
        myRNN = nn.GRU(input_size, hidden_size, batch_first=True)
    else:
        raise ValueError("Error of input cell_type, please choose one from [rnn, lstm, gru]")
    # move the kernel to GPU
    myRNN.cuda()
   
    # resul ops
    if compute_type=="forward":
        compfunc = run_forward
        
    elif compute_type=="backward":
        compfunc = run_backward
        
    elif compute_type=="calibrate":
        compfunc = run_calibrate
    else:
        raise ValueError("Error, compute_type should be either forward or backward or calibrate")
   

    print("warming up for {} steps".format(n_warm))
    start = time.time()
    for i in range(n_warm):
        compfunc(input_image, myRNN)
    end = time.time()
    print("done")
    duration = end-start
    print('Warmup {:.2f} seconds, {:.2f} seconds/iter'.format(duration, duration/float(n_warm)))
    
    print("running for {} steps".format(n_iter))
    start = time.time()
    if os.environ['PROFILER'] == 'pycuda':
        if have_pycuda:
            pyc.driver.start_profiler()
    elif os.environ['PROFILER'] == 'cupy':
        if have_cupy:
            cupy.cuda.profiler.start()
            
    for i in range(n_iter):
        compfunc(input_image, myRNN)
        
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
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', default=[10,32,32], help='the shape of the input tensor')
    AP.add_argument('--cell_type', type=str, default='lstm', help='the rnn cell type\
')
    AP.add_argument('--n_neurons', type=int, default=50, help='number of neurons for\
 the layer')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="forward", help='forward or backward pass')
    parsed = AP.parse_args()
    
    #print args
    for arg in vars(parsed):
        print(arg, ":", getattr(parsed, arg))
        
    
    main(input_tensor_shape=parsed.input_tensor_shape,
         cell_type=parsed.cell_type,
         n_neurons=parsed.n_neurons,
         dtype=parsed.dtype,
         n_iter=parsed.num_iterations,
         n_warm=parsed.num_warmups,
         compute_type=parsed.compute_type)
    


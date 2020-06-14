## a rnn1d kernel using tensorflow

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

warnings.simplefilter('ignore')
#tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = batch_first)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        r_out, _ = self.rnn(x)
        result  = self.linear(r_out)
        return result



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = batch_first)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        r_out, _ = self.lstm(x)
        result  = self.linear(r_out)
        return result


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = batch_first)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        r_out, _ = self.gru(x)
        result  = self.linear(r_out)
        return result


def run_calibrate(input_image, lstm, run_device):
    #with tf.device(gpu_dev):
    exec_op = input_image
    input_image = input_image.to(run_device)
    input_image.cpu().detach().numpy()


def run_forward(input_image, lstm, run_device):
#with tf.device(gpu_dev):
    lstm.to(run_device)
    input_image = input_image.to(run_device)
    exec_op = lstm(input_image)
    exec_op.cpu().detach().numpy()

    
def run_backward(input_image, lstm, run_device):

    lr = 0.01
    momentum = 0.5
    optimizer = optim.SGD(lstm.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()
        
      #with tf.device(gpu_dev):
    lstm.to(run_device)
    input_image = input_image.to(run_device)
    exec_op = lstm(input_image)
    exec_op.sum().backward()
    optimizer.step()
    exec_op.cpu().detach().numpy()

    
def main(input_tensor_shape, cell_type, input_size, hidden_size, num_layers, dtype, n_iter, n_warm, compute_type): #, enable_xla, agg_placement):

    if dtype == 'float16':
        tensor_type=torch.float16
    elif dtype == 'float32':
        tensor_type=torch.float32
    else:
        raise Exception('data type can only be float16 or float32')
    
#     if enable_xla:
#         gpu_dev = "cuda:0"
#         cpu_dev = "/device:XLA_CPU:0"
#     else:
#         gpu_dev = "cuda:0"
#         cpu_dev = "/device:CPU:0"
        

    #if agg_placement:
    #    agg_dev = cpu_dev
    #else:
    #    agg_dev = gpu_dev

    if torch.cuda.device_count():
        run_device = 'cuda:0'
    else:
        run_device = 'cpu'
        
    
    #with tf.device(agg_dev):
        #input tensor
        #input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=dtype) 
    input_image = np.random.randn(input_tensor_shape[0], input_tensor_shape[1], input_tensor_shape[2])
    input_image = torch.FloatTensor(input_image)    


    batch_first = True
    if cell_type == 'LSTM':
        lstm = LSTM(input_size, hidden_size, num_layers, batch_first)
    elif cell_type == 'RNN':
        lstm = RNN(input_size, hidden_size, num_layers, batch_first)
    elif cell_type == 'GRU':
        lstm = GRU(input_size, hidden_size, num_layers, batch_first)
    else:
        raise ValueError("Error of input cell_type, please choose one from [RNN, LSTM, GRU]")
   
    # resul ops
    if compute_type=="forward":
        compfunc = run_forward
        
    elif compute_type=="backward":
        compfunc = run_backward
        
    elif compute_type=="calibrate":
        compfunc = run_calibrate
    else:
        raise ValueError("Error, compute_type should be either forward or backward or calibrate")
   
    #start session
    #sess_config=tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
    #sess_config.gpu_options.allow_growth = True
    #with tf.Session(config=sess_config) as sess:
    #    sess.run(init_op)
        
    print("warming up for {} steps".format(n_warm))
    start = time.time()
    for i in range(n_warm):
        compfunc(input_image, lstm, run_device)
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
        compfunc(input_image, lstm, run_device)
        
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
    AP.add_argument('--cell_type', type=str, default='LSTM', help='the rnn cell type\
')
    AP.add_argument('--input_size', type=int, default=32, help='the input size for LSTM')
    AP.add_argument('--hidden_size', type=int, default=5, help='the hidden size of lstm')
    AP.add_argument('--num_layers', type=int, default=3, help='the number of layers of lstm')
    #AP.add_argument('--n_neurons', type=int, default=50, help='number of neurons forthe layer')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="forward", help='forward or backward pass')
#     AP.add_argument('--enable_xla', action="store_true", help="enable XLA support")
#     AP.add_argument('--aggressive_placement', action="store_true", help='if enabled, place everything which is not convolution on the CPU')
    parsed = AP.parse_args()
    
    #print args
    for arg in vars(parsed):
        print(arg, ":", getattr(parsed, arg))
        
    
    main(input_tensor_shape=parsed.input_tensor_shape,
         cell_type=parsed.cell_type,
         input_size = parsed.input_size,
         hidden_size=parsed.hidden_size,
         num_layers=parsed.num_layers,
         #n_neurons=parsed.n_neurons,
         dtype=parsed.dtype,
         n_iter=parsed.num_iterations,
         n_warm=parsed.num_warmups,
         compute_type=parsed.compute_type)
#     ,
#          enable_xla=parsed.enable_xla,
#          agg_placement=parsed.aggressive_placement)
    
    


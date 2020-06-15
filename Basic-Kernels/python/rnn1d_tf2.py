## A rnn1d kernel using tensorflow 2.2

import os
import warnings
# import tensorflow.compat.v1 as tf
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()
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
# tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("Eager execution: {}".format(tf.executing_eagerly()))


# @tf.function(experimental_compile=True)
def rnn1d(input_data, cell_type, n_neurons, dtype):
    if cell_type == 'rnn':
        basic_cell = tf.keras.layers.SimpleRNNCell(units=n_neurons)
    elif cell_type == 'lstm':
        basic_cell = tf.keras.layers.LSTMCell(units=n_neurons)
    elif cell_type == 'gru':
        basic_cell = tf.keras.layers.GRUCell(units=n_neurons)
    else:
        raise Exception("cell_type could only be: rnn, lstm or gru!")

    outputs, states = tf.keras.layers.RNN(basic_cell, return_sequences=True, return_state=True)(input_data)
    return outputs, states


#calibration measurement
def run_calibrate(input_tensor_shape, cell_type, n_neurons, tensor_type):
    input_image = tf.keras.backend.random_uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=tensor_type)
#     _ = input_image.numpy()
    return input_image


#forward
def run_forward(input_tensor_shape, cell_type, n_neurons, tensor_type):
    input_image = tf.keras.backend.random_uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=tensor_type)
    output_result, states_cur = rnn1d(input_image, cell_type, n_neurons, tensor_type) 
#     _,_ = output_result.numpy(), states_cur.numpy()
    return output_result


#backward
def run_backward(input_tensor_shape, cell_type, n_neurons, tensor_type):
    input_image = tf.keras.backend.random_uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=tensor_type)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_image)
        output_result, states_cur = rnn1d(input_image, cell_type, n_neurons, tensor_type)
    grads = tape.gradient(output_result, input_image)
#     tvars = tf.trainable_variables()
#     grads = tape.gradient(output_result,tvars)
#     opt = tf.keras.optimizers.SGD() 
#     opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

#     _ = grads.numpy()
    return grads


def main(input_tensor_shape, cell_type, n_neurons, dtype, n_iter, n_warm, compute_type, enable_xla):
    
    #datatype selection
    if dtype == 'float16':
        tensor_type=tf.float16
    elif dtype == 'float32':
        tensor_type=tf.float32
    else:
        raise Exception('data type can only be float16 or float32')
    tf.keras.backend.set_floatx(dtype)
#     tf.config.optimizer.set_jit(False)
    
    ##XLA or not
    if tf.config.list_physical_devices('GPU'):
        device = '/device:XLA_GPU:0' if enable_xla else '/GPU:0'
    else:
        device = '/device:XLA_CPU:0' if enable_xla else '/CPU:0'
        
    print("Running on device {}".format(device))
    print(tf.config.experimental.list_physical_devices())
    ##tf.config.gpu.set_per_process_memory_growth(True)
#     tf.config.set_soft_device_placement(False)
    tf.debugging.set_log_device_placement(True)
#     print(tf.config.get_soft_device_placement())
#     tf.config.experimental.set_memory_growth(device,True)


    # select commpute type
    if compute_type == "forward":
        compfunc = run_forward
    elif compute_type == "backward":
        compfunc = run_backward
    elif compute_type == "calibrate":
        compfunc = run_calibrate
    else:
        raise ValueError("Error, compute_type should be either forward or backward or calibrate")
    

    #start session
    print("warming up for {} steps".format(n_warm))
    start = time.time()
    with tf.device(device):
        for i in range(n_warm):
            compfunc(input_tensor_shape, cell_type, n_neurons, tensor_type)
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
            
    with tf.device(device):
        for i in range(n_iter):
            compfunc(input_tensor_shape, cell_type, n_neurons, tensor_type)

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
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', default=[10,32,32], help='the shape of the input tensor, timesteps x batchsize x celldepth')
    AP.add_argument('--cell_type', type=str, default='lstm', help='the rnn cell type\
')
    AP.add_argument('--n_neurons', type=int, default=50, help='number of neurons for\
 the layer')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="forward", help='forward or backward pass')
    AP.add_argument('--enable_xla', action="store_true", help="enable XLA support")
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
         compute_type=parsed.compute_type,
         enable_xla=parsed.enable_xla)
    
    


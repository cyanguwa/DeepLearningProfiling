## a conv2d kernel using tensorflow

import os
import warnings
import tensorflow as tf
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
else:
    print('Please make sure Start/Stop profiler is installed')

warnings.simplefilter('ignore')
#tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


#N = 10
#C = 3
#H = 128
#W = 128

#input_tensor_shape = [10,128,128,3]  # NHWC
#kernel_shape = [5,5,3,10]
#stride_ = [1,2,2,1]

def conv2d(input_data, data_format, kernel_shape, stride_, dtype):
    weights = tf.Variable(tf.random.truncated_normal(kernel_shape, stddev=0.03, dtype=dtype), dtype=dtype)
    if data_format == "NCHW":
        input_data = tf.transpose(input_data, [0,3,1,2])
        strides = [1,1,stride_, stride_]
    else:
        strides = [1,stride_,stride_,1]
    output_data = tf.nn.conv2d(input_data, weights, strides=strides, padding='SAME', data_format=data_format)
    return output_data


def main(input_tensor_shape, data_format, kernel_shape, stride, dtype, n_iter, n_warm, compute_type, enable_xla, agg_placement):

    if dtype == 'float16':
        tensor_type=tf.float16
    elif dtype == 'float32':
        tensor_type=tf.float32
    else:
        raise Exception('data type can only be float16 or float32')
    
    if enable_xla:
        gpu_dev = "/device:XLA_GPU:0"
        cpu_dev = "/device:XLA_CPU:0"
    else:
        gpu_dev = "/device:GPU:0"
        cpu_dev = "/device:CPU:0"
        
    if agg_placement:
        agg_dev = cpu_dev
    else:
        agg_dev = gpu_dev

    with tf.device(agg_dev):
        #input tensor
        input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=dtype) 
        
    with tf.device(gpu_dev):
        
        #create network
        output_result = conv2d(input_image, data_format, kernel_shape, stride, tensor_type) 
        
        #init ops
        init_op = tf.compat.v1.initializers.global_variables()
        
    #resul ops
    if compute_type=="forward":
        with tf.device(gpu_dev):
            exec_op = output_result
    elif compute_type=="backward":
        with tf.device(gpu_dev):
            opt = tf.compat.v1.train.GradientDescentOptimizer(0.5)
            exec_op = opt.compute_gradients(output_result)
#             grads = opt.compute_gradients(output_result)
#             exec_op = opt.apply_gradients(grads)
#             exec_op = opt.minimize(output_result)

    elif compute_type=="calibrate":
        with tf.device(gpu_dev):
            exec_op = input_image
    else:
        raise ValueError("Error, compute_type should be either forward or backward or calibrate")
   
    #start session
    sess_config=tf.compat.v1.ConfigProto(allow_soft_placement=False,log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
#    print(tf.config.experimental.list_physical_devices())

    with tf.compat.v1.Session(config=sess_config) as sess:
        sess.run(init_op)
        
        print("warming up for {} steps".format(n_warm))
        start = time.time()
        for i in range(n_warm):
            result = sess.run(exec_op)
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
            result = sess.run(exec_op)
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
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', help='the shape of the input tensor. Note that it depends on data_format (default NHWC)')
    AP.add_argument('--data_format', type=str, default='NHWC', help='choose either channels_last or channels_first')
    AP.add_argument('--kernel_shape', type=int, nargs='+', default=[5,5,3,32], help='the shape of the conv kernel [filter_height, filter_width, in_channels, out_channels]')
    AP.add_argument('--stride', type=int, default=1, help='the stride')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="forward", help='forward or backward pass')
    AP.add_argument('--enable_xla', action="store_true", help="enable XLA support")
    AP.add_argument('--aggressive_placement', action="store_true", help='if enabled, place everything which is not convolution on the CPU')
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
         compute_type=parsed.compute_type,
         enable_xla=parsed.enable_xla,
         agg_placement=parsed.aggressive_placement)
    
    


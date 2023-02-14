import sys
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

BATCH_SIZE = 1

class DarknetTrt():
    def __init__(self, opt):
        target_dtype = np.float16

        f = open(opt.weights, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

        # need to set input and output precisions to FP16 to fully enable it
        self.output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 

        # allocate device memory
        self.d_input = cuda.mem_alloc(1 * opt.img_size[0] * opt.img_size[1] * sys.getsizeof(target_dtype))
        self.d_output = cuda.mem_alloc(1 * output.nbytes)

        self.bindings = [int(d_input), int(d_output)]

        self.stream = cuda.Stream()

    def __call__(batch):
        # transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # syncronize threads
        self.stream.synchronize()
        
        return self.output

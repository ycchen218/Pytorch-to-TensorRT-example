import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import time
import psutil
import matplotlib.pyplot as plt

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """
      設定每個執行緒的輸入形狀、資料類型、名稱
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        # print("size: ",size)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # print("dtype: ",dtype)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        # print("host_mem: ",host_mem)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # print("device_mem: ",device_mem)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    print(inputs)
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
  """
    1. cuda.memcpy_htod_async：此步驟將輸入數據從主機（CPU）異步傳輸到GPU。它會在輸入數據準備就緒後開始執行異步傳輸，這意味著它會繼續執行後續代碼，而不會等待數據傳輸完成。

    2. context.execute_async：此步驟執行TensorRT引擎的推理。它將觸發異步的推理操作，TensorRT引擎開始執行模型，進行預測，而不會阻塞主機的執行。可以在數據傳輸到GPU的同時進行推理，以充分利用硬件資源。

    3. cuda.memcpy_dtoh_async：此步驟將輸出數據從GPU異步傳輸回主機。與輸入類似，它也是異步執行的，不會阻塞主機的執行。這使得可以在數據傳輸回主機的同時執行其他操作。

    4. stream.synchronize()：此步驟是同步CUDA流，確保之前的異步操作已經完成。在這裡，主機程序會等待直到所有異步傳輸和推理操作都完成。這是為了確保在訪問輸出數據之前，數據已經準備好並可用。
  """
  # Transfer input data to the GPU.
  [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
  # Run inference.
  context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
  # Transfer predictions back from the GPU.
  [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
  # Synchronize the stream
  stream.synchronize()
  # Return only the host outputs.
  return [out.host for out in outputs]

image_path = "6.png"
input_shape = (28, 28)

image = cv2.imread(image_path)
image = cv2.resize(image, input_shape)
image = image.astype(np.float32)
image /= 255.0
plt.imshow(image)
plt.show()
image = np.transpose(image,(2,0,1))
image = image[np.newaxis,...]
image = np.ascontiguousarray(image)
print(image.shape)


with open("mnist_model_FP16.trt", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

context = engine.create_execution_context()


inputs, outputs, bindings, stream = allocate_buffers(engine)
inputs[0].host = image.ravel()

start_time = time.time()

with engine.create_execution_context() as context:
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)


end_time = time.time()
execution_time = end_time - start_time


gpu_usage = psutil.virtual_memory().percent
cpu_usage = psutil.cpu_percent()
ram_usage = psutil.virtual_memory().used

predicted_class = np.argmax(trt_outputs[0])
print("Predicted class:", predicted_class)
print("Execution Time:", execution_time, "seconds")
# print("GPU Usage:", gpu_usage, "%")
# print("CPU Usage:", cpu_usage, "%")
# print("RAM Usage:", ram_usage, "bytes")
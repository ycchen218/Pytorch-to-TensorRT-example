import torch
import torch.nn as nn
from data import makeData
from train import modelTrain
from model import MNIST_ResNet34
import tensorrt as trt

def pt2Onnx(pt_file,onnx_file):
    model = MNIST_ResNet34()
    model.load_state_dict(torch.load(f"{pt_file}.pt"))
    model.eval()

    dummy_input = torch.randn(1, 3, 28, 28)

    onnx_path = f'{onnx_file}.onnx'

    torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
    print("Finish pt to onnx")

def Onnx2Trt(onnx_path,trt_path):
    # Load the ONNX model
    onnx_model_path = f"{onnx_path}.onnx"
    trt_model_path = f"{trt_path}.trt"

    # Create a TensorRT builder and network
    trt_logger = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network,
                                                                                                               trt_logger) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28
        builder.max_batch_size = 1

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            trt_model_path = f"{trt_path}_FP16.trt"
            print("In FP16!!!")
        else:
            print("Warning: Fast FP16 not supported on this platform. Using default precision.")


        with open(onnx_model_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        print('Completed parsing of ONNX file')
        plan = builder.build_serialized_network(network, config)
        with trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(trt_model_path, "wb") as f:
            f.write(engine.serialize())







if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # 定義 單一批次個數
    batch_size = 32
    # 定義 訓練次數
    epochs = 5
    # 建立 DataLoader
    train_loader, test_loader = makeData(batch_size=batch_size)
    # 建立 model
    model = MNIST_ResNet34().to(device)
    # 定義 損失函數
    loss_fn = nn.CrossEntropyLoss()
    # 定義 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    #pytorch 訓練並儲存模型
    modelTrain(epochs=epochs, train_loader=train_loader, test_loader=test_loader, model=model, loss_fn=loss_fn,
               opt=optimizer, device=device, save_file='mnist_model')

    # .pt檔 轉 .onnx
    pt2Onnx(pt_file='mnist_model',onnx_file='mnist_model')

    # .onnx 轉 .trt
    Onnx2Trt(onnx_path='mnist_model',trt_path='mnist_model')









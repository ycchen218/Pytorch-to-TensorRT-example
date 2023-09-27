# Pytorch-to-TensorRT-example
![image](https://github.com/ycchen218/Pytorch-to-TensorRT-example/blob/master/git_image/structure.jpg)
## Introduce
**All in Python** <br>
This is an MNIST example demonstrating how to convert a .pt file to an .ONNX file, and subsequently transform the .ONNX file into a .TRT file. Additionally, it illustrates how to save the .TRT file in FP16 mode, which can reduce memory usage and accelerate computation. Importantly, I also provide TensorRT inference code for reference. This code can assist you in efficiently performing inference using the created .TRT file, thereby enhancing your understanding of the process.
## Requirement
1. python3.8
2. tensorrt
3. matplotlib
4. numpy
5. opencv
6. pytorch 1.12.0
7. torchvision
## Execute
```markdown
python main.py
```
The main.py file incude function as following:
1. Train the mnist classifier ans save the model in .pt file.
2. Convert the .pt file to .onnx file.
3. Convert the .onnx file to .trt file.
## Inference the .trt model
```markdown
python test_trt.py
```
Run the test_trt.py to Inference the mnist image prediction result.
## Test Torch model
```markdown
python test_pt.py
```
Run the test_pt.py to check the saved .pt model weight prediction result. 


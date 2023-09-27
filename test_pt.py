import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from model import MNIST_ResNet34

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

image_path = r"5.png"
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
image = torch.Tensor(image)


model = MNIST_ResNet34().to(device)
model_weights_path = "mnist_model.pt"
model.load_state_dict(torch.load(model_weights_path, map_location=device))


model.eval()

pred = model(image.to(device))
pred = pred.cpu().detach().numpy()
print(pred)
print(np.argmax(pred))
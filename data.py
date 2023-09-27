import torch
from torchvision import datasets, transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def makeData(batch_size=32):
    # 資料轉換函數
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

    # 建立 MNIST 的 Dataset
    mnist_train = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    mnist_test = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    print("訓練資料集數量：", len(mnist_train))
    print("測試資料集數量：", len(mnist_test))


    # 建立 DataLoader
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)

    return train_loader,test_loader
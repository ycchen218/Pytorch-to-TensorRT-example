import torch
import numpy as np
import cv2



def train(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)


        # x_numpy = X[0].cpu().numpy()
        # x_numpy = np.transpose(x_numpy,(1,2,0))
        # plt.imshow(x_numpy)
        # plt.show()

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn,device):
    size = len(dataloader.dataset)

    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def modelTrain(epochs,train_loader,test_loader,model,loss_fn,opt,device,save_file):
    # in_shape = (bs,3,28,28)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, opt,device)
        test(test_loader, model, loss_fn,device)
    torch.save(model.state_dict(), f'{save_file}.pt')
    print("完成！")








import torch
import torch.nn as nn
import visdom
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ae import AutoEncoder


def main():
    mnist_train = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()]
    ))
    mnist_test = datasets.MNIST('data/', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()]
    ))

    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

    x, y = iter(train_loader).next()
    print(x.shape)

    device = torch.device('cpu')
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    vis = visdom.Visdom()
    for epoch in range(1000):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())
        x, y = iter(test_loader).next()
        with torch.no_grad():
            x_hat = model(x)
        vis.images(x, nrow=8, win='x-ae', opts=dict(title='x'))
        vis.images(x_hat, nrow=8, win='x-hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()

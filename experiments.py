import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from dataset import SiameseDataset
from loss import ParwiseLoss
from network import Net
from train import *


def plot2D(data):
    plt.title('Dispercao  embedings 3D')
    plt.axes(projection='3d')

    colors = ['blue', 'red', 'black', 'pink', 'green', 'yellow', 'orange', 'gray', 'brown', 'darkblue']
    for x, y, z, t in data:
        plt.scatter(x, y, z, c=colors[int(t)])

    plt.savefig('imgs/3Dplot.png')
    plt.close()


def plotLoss(epochs, train_losses, test_losses):
    plt.title('Grafico Loss')
    plt.xlabel('iteration')
    plt.ylabel('Loss value')

    plt.plot(range(epochs), train_losses)
    plt.plot(range(epochs), test_losses)
    plt.legend(['train', 'validation'])
    plt.savefig('imgs/loss.png')
    plt.close()


def main():

    BATCH_SIZE = 100
    LR = 1e-3
    EPOCHS = 15

    print('Number of epochs: ', EPOCHS)
    print('Learning Rate: ', LR)
    print('Batch size: ', BATCH_SIZE)

    transforms_ = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    mnist_train = datasets.FashionMNIST('../data', train=True, download=True, transform=transforms_)
    dataset_train = SiameseDataset(mnist_train)
    sTrainDataLoader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)

    mnist_test = datasets.FashionMNIST('../data', train=False, download=True, transform=transforms_)
    dataset_test = SiameseDataset(mnist_test)
    sTestDataLoader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    pairwise_loss = ParwiseLoss()
    net = Net()
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.1,
                                                     patience=1,
                                                     verbose=True)
    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        print('Epoch:{} '.format(epoch))
        train_loss = train(net, optimizer, sTrainDataLoader, pairwise_loss)
        test_loss = test(net, sTestDataLoader, pairwise_loss)
        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    sTestDataLoader2 = DataLoader(dataset_test,
                                  batch_size=10000,
                                  shuffle=False, num_workers=6)
    emb = getembeddings(net, sTestDataLoader2)

    plotLoss(EPOCHS, train_losses, test_losses)
    plot2D(emb)


if __name__ == '__main__':
    main()

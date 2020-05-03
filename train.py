import torch
import numpy as np
import time


def train(net_, opti_, data_loader_, pairwise_loss_):
    acc = 0
    tic = time.time()
    for idx, data in enumerate(data_loader_):
        d1, d2, y = data[0].cuda(), data[1].cuda(), data[2]
        opti_.zero_grad()
        out, out1 = net_(d1, d2)
        loss_v = pairwise_loss_(out, out1, y)
        acc += loss_v.data.item()
        loss_v.backward()
        opti_.step()

    toc = time.time()
    print('Training time: {:.2f}'.format(toc - tic))
    mean_loss = acc / len(data_loader_)
    print('Mean train loss: {:.3f}'.format(mean_loss))
    return mean_loss


def test(net_, data_loader_, parwise_loss_):
    net_.eval()
    acc = 0
    with torch.no_grad():
        for idx, data in enumerate(data_loader_):
            d1, d2, y = data[0].cuda(), data[1].cuda(), data[2]
            output1, output2 = net_(d1, d2)
            acc += parwise_loss_(output1, output2, y)

        test_loss = acc / len(data_loader_)
        print('Mean test loss: {:.3f}\n'.format(test_loss))
    return test_loss


def getembeddings(net_, data_loader):
    embs = []
    labels = []
    with torch.no_grad():
        net_.eval()
        for idx, data in enumerate(data_loader):
            emb = net_.forward_once(data[0].cuda())
            embs = np.stack(emb.cpu().numpy())
            labels = (np.stack(data[3].cpu().numpy()))

        labels = labels.reshape((labels.shape[0], 1))
        embs = np.hstack((embs, labels))

    return embs

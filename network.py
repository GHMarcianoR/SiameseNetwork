import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 30, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(30, 40, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(40, 40, 3),
                                   nn.ReLU(),
                                   nn.Conv2d(40, 60, 3),
                                   nn.Dropout2d(0.3),
                                   nn.ReLU(),
                                   nn.Conv2d(60, 50, 2),
                                   nn.ReLU(),
                                   nn.Conv2d(50, 40, 2),
                                   nn.Dropout2d(0.3),
                                   nn.ReLU(),
                                   nn.Conv2d(40, 20, 2),
                                   nn.ReLU()
                                   )

        self.fc1 = nn.Sequential(nn.Linear(20 * 17 * 17, 2000),
                                 nn.ReLU(),
                                 nn.Linear(2000, 500),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.ReLU(),
                                 nn.Linear(500, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, 100),
                                 nn.Dropout(0.3),
                                 nn.ReLU(),
                                 nn.Linear(100, 3),
                                 )

    def forward_once(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        inp1 = self.forward_once(input1)
        inp2 = self.forward_once(input2)
        return inp1, inp2

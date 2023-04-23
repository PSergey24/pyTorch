import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm


class NeuralNumbers(nn.Module):
    def __init__(self):
        super().__init__()

        self.flat = nn.Flatten()
        self.linear_1 = nn.Linear(28*28, 100)
        self.linear_2 = nn.Linear(100, 10)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.flat(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


def main():
    trans = tv.transforms.Compose([
        tv.transforms.ToTensor()
    ])

    ds_mnist = tv.datasets.MNIST('./datasets', download=True, transform=trans)

    # display picture
    # plt.imshow(ds_mnist[0][0].numpy()[0])

    batch_size = 16
    dataloader = DataLoader(ds_mnist, batch_size=batch_size, shuffle=True,
                            num_workers=1, drop_last=True)

    model = NeuralNumbers()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    epochs = 10
    for epoch in range(epochs):
        loss_val = 0
        for img, label in tqdm(dataloader):
            optimizer.zero_grad()

            label = F.one_hot(label, 10).float()
            prediction = model(img)

            loss = loss_fn(prediction, label)

            loss.backward()
            loss_val += loss.item()

            optimizer.step()


def accuracy(prediction, label):
    answer = F.softmax(prediction.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()

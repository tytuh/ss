import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# Create dummy dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    output = net(torch.from_numpy(X_train).float())
    loss = criterion(output, torch.from_numpy(y_train).long())
    loss.backward()
    optimizer.step()

# Test the network
output = net(torch.from_numpy(X_test).float())
_, predicted = torch.max(output.data, 1)
acc = accuracy_score(y_test, predicted.numpy())

st.write("Accuracy: ", acc)
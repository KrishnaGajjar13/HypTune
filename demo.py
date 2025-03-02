import torch, optuna
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
from sklearn.model_selection import train_test_split

print("Using " + torch.__version__ + " version of torch")
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


df = pd.read_csv('/content/Fashion_MNIST/fashion-mnist_train.csv')
X = df.iloc[:,1:].values
y = df.iloc[:,0].values

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.25, random_state=69)
X_train = X_train/255.0
x_test = X_test/255.0

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, features, labels):
    self.features = torch.tensor(features , dtype = torch.float32)
    self.labels = torch.tensor(labels, dtype=torch.long)
  def __len__(self):
    return len(self.features)
  def __getitem__(self, index):
    return self.features[index], self.labels[index]

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory= True)
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory= True)

class MyNN(nn.Module):
  def __init__(self,input_dims,output_dim, num_hidden_layers, neurons_per_layer,dropout_prob,layer_reduction_rate):
    super().__init__()

    layers = []

    for i in range(num_hidden_layers):
      if input_dims < 2 * output_dim:
         break
      layers.append(nn.Linear(input_dims,neurons_per_layer))
      layers.append(nn.BatchNorm1d(neurons_per_layer))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(p=dropout_prob))
      input_dims = neurons_per_layer * (1-layer_reduction_rate)

    layers.append(nn.Linear(input_dims,output_dim))

    self.model = nn.Sequential(*layers)

  def forward(self,x):
    return self.model(x)

def objective(trial):
    #State Search
    num_hidden_layers =   trial.suggest_int("num_hidden_layers" ,1,5)
    neurons_per_layer = trial.suggest_int("neurons_per_layer" ,64,256,step = 16)
    dropout_prob = trial.suggest_float("dropout_prob" ,0.1,0.3)
    layer_reduction_rate = trial.suggest_float("layer_reduction_rate" ,0.1,0.4, step = 0.1)
    #model init
    input_dim = 784
    output_dim = 10
    model = MyNN(input_dim,output_dim, num_hidden_layers, neurons_per_layer,dropout_prob,layer_reduction_rate)

    #loss function
    criterion = nn.CrossEntropyLoss()

    #params init
    learning_rate = 0.01
    epochs = 10

    #optimizer selection
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,weight_decay = 1e-4)


    #training loop
    for epoch in range(epochs):
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device) # Move data to device
            preds = model(data)

            loss = criterion(preds,targets)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
    #evaluation
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for data, targets in test_loader:
        # data, targets = data.to(device), targets.to(device) # Move data to device
            preds = model(data)

            _, predicted = torch.max(preds.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            accuracy = 100*correct/total

    return accuracy

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trial = 50)
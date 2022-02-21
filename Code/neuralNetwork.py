import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import CustomDataset
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential( #nota futura: nº layers = 2, nº nodes = 75% do input
            nn.Linear(515, 385),
            nn.ReLU(),
            nn.Linear(385, 290),
            nn.ReLU(),
            nn.Linear(290, 1),
            nn.Sigmoid(),#necessary for bceloss
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y, z) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device) #moving to device(GPU)
        
        pred = model(X.float()).squeeze(1)
        print(pred)
        
        loss = loss_fn(pred, (y.float()))
        # Backpropagation
        
        optimizer.zero_grad()
        loss.backward()
        ##check weights being updated 
        # print("Model structure: ", model, "\n\n")
        # for name, param in model.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        optimizer.step()

        if batch % 1 == 0: #progress update
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, z in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X.float()).squeeze(1)
            print("Predicton: ", end=' ')
            print(pred.item(), end=' ')
            print("label: ", end=' ')
            print(y.item(), end=' ')
            print("File name: ", end=' ')
            print(z)

            
            test_loss += loss_fn(pred, (y.float()))
            
            test = (pred > 0.5).int()
            
            correct += torch.eq(test, y).sum().float()
            
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    


#--------------------------------------------Paths,GPU,Inicialization--------------------------------------------#  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ' + device + ' device') 

current_dir = os.getcwd()

data_dir = current_dir + '/data'
annotations_file = data_dir + '/labels.csv'

badbox_dir = current_dir + '/bad_gearboxes'
badbox_annotations_file = badbox_dir + '/labels.csv'


model = NeuralNetwork().to(device)

#model.load_state_dict(torch.load(current_dir + '/network_saves/save.pt'))

#--------------------------------------------Parameters--------------------------------------------#

#Hyperparameters
learning_rate = 1e-4
batch_size = 1
epochs = 20

#Loss Function
loss_fn = nn.BCELoss() #BCE precisa de uma funcçao sigmoid na ultima layer
#loss_fn = nn.CrossEntropyLoss() #Já tem softmax embutido


#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #Stochastic Gradient Descent

#--------------------------------------------datasets n dataloader--------------------------------------#
full_dataset = CustomDataset(annotations_file, data_dir)
badbox_dataset = CustomDataset(badbox_annotations_file, badbox_dir)

train_size = int(0.8 * len(badbox_dataset))
test_size = len(badbox_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(badbox_dataset, [train_size, test_size])

badbox_dataloader = DataLoader(badbox_dataset, batch_size, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

# train_features, train_labels = next(iter(train_dataloader))
# print("Feature batch shape: {"  + str(train_features.size()) + "}")
# print("Labels batch shape: {" + str(train_labels.size()) + "}")

# train_features = train_features.to(device)
#--------------------------------------------Neural Network --------------------------------------------#

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    
    train_loop(train_dataloader, model, loss_fn, optimizer)
    print("test")
    test_loop(test_dataloader, model, loss_fn)
    
    
print("Done!")


torch.save(model.state_dict(), current_dir + '/network_saves/save.pt')








# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])


# #print(model)
# X = train_features
# logits = model(X.float())
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")


from datetime import datetime

from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
import torch


class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim, num_classes):
    super(LinearClassifier, self).__init__()

    self.linear = torch.nn.Linear(input_dim, num_classes)

  def forward(self, x):
    x = self.linear(x)
    return x

# TODO: Create a 'BatchGenerator' for training, validation and test datasets.
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])
p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",1)
training_Batches=BatchGenerator(p,500,False,op)
p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",2)
validation_Batches=BatchGenerator(p,500,False,op)
p=PetsDataset(r"C:\Users\marti\PycharmProjects\pythonProject\cifar-10-batches-py",3)
test_Batches=BatchGenerator(p,500,False,op)




# TODO: Create the LinearClassifier, loss function and optimizer.
model = LinearClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


'''
TODO: Train a model for multiple epochs, measure the classification accuracy on the validation dataset throughout the training and save the best performing model. 
After training, measure the classification accuracy of the best perfroming model on the test dataset. Document your findings in the report.


'''

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

best_vloss = 1_000_000.
n_epochs=100
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for it in range(n_epochs):
    running_loss = 0.
    last_loss = 0.
    for local_batch, local_labels in training_Batches:
        local_batch=torch.from_numpy(local_batch)
        local_labels=torch.from_numpy(local_labels)
        # Transfer to GPU
        # local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        optimizer.zero_grad()
        outputs = model(local_batch)
        # Compute the loss and its gradients
        loss = criterion(outputs, local_labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()

        if (it+1) % 10 ==0:
            model.train(False)
            for i,vlocal_batch, vlocal_labels in validation_Batches:

                voutputs = model(vlocal_batch)
                vloss = criterion(voutputs, vlocal_labels)
                print(f'In this epoch {it+1}/{n_epochs}, Training loss: {loss.item():.4f}, Test loss: {vloss.item():.4f}')

                # Track best performance, and save the model's state
                if vloss.item() < best_vloss:
                    best_vloss = vloss.item()
                    model_path = 'model_{}_{}'.format(timestamp, it)
                    torch.save(model.state_dict(), model_path)





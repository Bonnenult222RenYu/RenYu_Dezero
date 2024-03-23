# Add import path for the dezero directory.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
import time

# Hyperparameters
max_epoch = 3
batch_size = 100
hidden_size = 1000


train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size, shuffle=True)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start
    print('epoch: {}, loss: {:.4f}, time: {:.4f}[sec]'.format(
        epoch + 1, sum_loss / len(train_set), elapsed_time))
    
model.save_weights('my_mlp.npz')
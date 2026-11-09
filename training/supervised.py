import torch
from torch import nn
from torch.utils.data import DataLoader
from models.encoder import ConvEncoder
from datasets.cifar10 import CIFAR10
from utils.checkpoint import save_checkpoint


def train(epochs=1,batch_size=64,limit=256,data='data',checkpoint='checkpoints/supervised.pt',device='cpu'):
    ds=CIFAR10(data,train=True,download=True,limit=limit); loader=DataLoader(ds,batch_size=batch_size,shuffle=True)
    model=nn.Sequential(ConvEncoder(),nn.Linear(512,10)).to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); fn=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total=0
        for x,y in loader:
            loss=fn(model(x.to(device)),y.to(device)); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()
        print(f'supervised epoch {epoch+1}/{epochs}: loss={total/max(1,len(loader)):.4f}')
    save_checkpoint(checkpoint,model,opt,epochs); return model

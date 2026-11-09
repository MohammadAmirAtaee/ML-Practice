import torch
from torch import nn
from torch.utils.data import DataLoader
from models.simsiam import SimSiam
from datasets.cifar10 import CIFAR10
from utils.checkpoint import load_checkpoint, save_checkpoint


def train(epochs=1, batch_size=64, limit=256, data='data', encoder_checkpoint='checkpoints/simsiam.pt', checkpoint='checkpoints/linear.pt', device='cpu'):
    ds=CIFAR10(data, train=True, download=True, limit=limit); loader=DataLoader(ds,batch_size=batch_size,shuffle=True)
    base=SimSiam().to(device); load_checkpoint(encoder_checkpoint,base,device=device); base.eval()
    for p in base.parameters(): p.requires_grad=False
    classifier=nn.Linear(512,10).to(device); opt=torch.optim.Adam(classifier.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total=0
        for x,y in loader:
            with torch.no_grad(): z=base.encode(x.to(device))
            loss=loss_fn(classifier(z),y.to(device)); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()
        print(f'linear epoch {epoch+1}/{epochs}: loss={total/max(1,len(loader)):.4f}')
    save_checkpoint(checkpoint, classifier, opt, epochs, encoder_checkpoint=encoder_checkpoint); return classifier

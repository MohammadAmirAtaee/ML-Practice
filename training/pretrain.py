import torch
from torch.utils.data import DataLoader
from models.simsiam import SimSiam
from datasets.cifar10 import CIFAR10
from utils.checkpoint import save_checkpoint


def train(epochs=1, batch_size=64, limit=256, data='data', checkpoint='checkpoints/simsiam.pt', device='cpu'):
    ds = CIFAR10(data, train=True, download=True, two_views=True, limit=limit); loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = SimSiam().to(device); opt = torch.optim.Adam(model.parameters(), lr=1e-3); history=[]
    model.train()
    for epoch in range(epochs):
        total=0
        for x1,x2,_ in loader:
            opt.zero_grad(); loss=model(x1.to(device),x2.to(device)); loss.backward(); opt.step(); total += loss.item()
        history.append(total/max(1,len(loader))); print(f'pretrain epoch {epoch+1}/{epochs}: loss={history[-1]:.4f}')
    save_checkpoint(checkpoint, model, opt, epochs); return model, history

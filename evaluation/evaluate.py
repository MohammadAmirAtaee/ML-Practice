import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.simsiam import SimSiam
from models.encoder import ConvEncoder
from datasets.cifar10 import CIFAR10, CLASSES
from utils.checkpoint import load_checkpoint


def evaluate_linear(checkpoint='checkpoints/linear.pt', encoder_checkpoint='checkpoints/simsiam.pt', data='data', batch_size=128, limit=None, device='cpu'):
    encoder = SimSiam().to(device); load_checkpoint(encoder_checkpoint, encoder, device=device); encoder.eval()
    classifier = nn.Linear(512, 10).to(device); load_checkpoint(checkpoint, classifier, device=device); classifier.eval()
    loader = DataLoader(CIFAR10(data, train=False, download=False, limit=limit), batch_size=batch_size)
    correct = total = 0; per_class = [0] * 10; per_total = [0] * 10
    with torch.no_grad():
        for x, y in loader:
            pred = classifier(encoder.encode(x.to(device))).argmax(1).cpu()
            correct += int((pred == y).sum()); total += len(y)
            for target, guess in zip(y.tolist(), pred.tolist()):
                per_total[target] += 1; per_class[target] += int(target == guess)
    result = {'accuracy': correct / max(1, total), 'correct': correct, 'total': total,
              'per_class_accuracy': {CLASSES[i]: per_class[i] / max(1, per_total[i]) for i in range(10)}}
    return result


def evaluate_supervised(checkpoint='checkpoints/supervised.pt', data='data', batch_size=128, limit=None, device='cpu'):
    model = nn.Sequential(ConvEncoder(), nn.Linear(512, 10)).to(device); load_checkpoint(checkpoint, model, device=device); model.eval()
    loader = DataLoader(CIFAR10(data, train=False, download=False, limit=limit), batch_size=batch_size)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += int((model(x.to(device)).argmax(1).cpu() == y).sum()); total += len(y)
    return {'accuracy': correct / max(1, total), 'correct': correct, 'total': total}


def save_report(report, path='results/evaluation.json'):
    Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps(report, indent=2), encoding='utf-8')

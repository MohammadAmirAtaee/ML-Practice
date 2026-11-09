from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer=None, epoch=0, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None, 'epoch': epoch, **extra}, path)


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    if not Path(path).exists(): raise FileNotFoundError(f'Checkpoint not found: {path}. Train a model first.')
    state = torch.load(path, map_location=device, weights_only=False); model.load_state_dict(state['model'])
    if optimizer and state.get('optimizer'): optimizer.load_state_dict(state['optimizer'])
    return state

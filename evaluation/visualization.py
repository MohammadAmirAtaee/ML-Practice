from pathlib import Path
import matplotlib.pyplot as plt


def save_curves(history, path='results/training_curves.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True); plt.figure(figsize=(7,4))
    for key, values in history.items(): plt.plot(values, label=key)
    plt.legend(); plt.xlabel('epoch'); plt.tight_layout(); plt.savefig(path); plt.close()

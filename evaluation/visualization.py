from pathlib import Path
import matplotlib.pyplot as plt


def save_curves(history, path='results/training_curves.png'):
    Path(path).parent.mkdir(parents=True, exist_ok=True); plt.figure(figsize=(7,4))
    for key, values in history.items(): plt.plot(values, label=key)
    plt.legend(); plt.xlabel('epoch'); plt.tight_layout(); plt.savefig(path); plt.close()


def save_accuracy_comparison(report, path='results/accuracy_comparison.png'):
    """Save a bar chart comparing evaluation accuracy values."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    names = ['SimSiam + Linear', 'Supervised Baseline']
    values = [report['linear']['accuracy'] * 100, report['supervised']['accuracy'] * 100]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=['#2878b5', '#55a868'], width=0.55)
    ax.set_ylim(0, 100); ax.set_ylabel('Accuracy (%)'); ax.set_title('CIFAR-10 Model Accuracy Comparison')
    ax.grid(axis='y', alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f'{value:.2f}%', ha='center', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

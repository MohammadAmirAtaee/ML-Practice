# SimSiam CIFAR-10 Project

An educational, modular implementation of SimSiam self-supervised learning on CIFAR-10. The project trains an encoder without labels, evaluates its representation with a linear classifier, compares it with a supervised baseline, and provides a local Gradio image-classification demo.

## Included

- SimSiam encoder, projector, predictor, stop-gradient, and cosine loss
- CIFAR-10 loader without `torchvision`
- CPU-compatible pretraining and evaluation
- Linear evaluation and supervised baseline
- Accuracy report in `results/evaluation.json`
- Gradio upload demo with class probabilities
- Guided notebook: `notebooks/experiment.ipynb`

## Installation (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Dataset

Download automatically:

```powershell
python main.py download
```

Or extract the dataset manually so this layout exists:

```text
data/cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

CIFAR-10 contains 32x32 RGB images in: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Training

Quick CPU smoke test:

```powershell
python main.py train-pretrain --epochs 1 --limit 256
python main.py train-linear --epochs 1 --limit 256
python main.py train-supervised --epochs 1 --limit 256
```

Use all 50,000 training images:

```powershell
python main.py train-pretrain --epochs 5 --limit 50000
python main.py train-linear --epochs 5 --limit 50000
python main.py train-supervised --epochs 5 --limit 50000
```

These commands create `checkpoints/simsiam.pt`, `checkpoints/linear.pt`, and `checkpoints/supervised.pt`. Increase `--epochs` for better results and longer training.

## Evaluation

Evaluate both models on all 10,000 test images:

```powershell
python main.py evaluate --limit 10000
```

The report is saved to `results/evaluation.json`.

## Demo

After pretraining and linear evaluation:

```powershell
python main.py demo
```

Open `http://127.0.0.1:7860`. Upload an image to see the image, predicted CIFAR-10 class, and probabilities for all ten classes. The demo is local and does not require internet. Images outside CIFAR-10, such as sheep or wool, are not expected to be classified correctly because those classes do not exist in the dataset.

## Jupyter notebook

Open `notebooks/experiment.ipynb` after installation. It demonstrates the architecture test, dataset loading, pretraining, linear evaluation, supervised comparison, checkpoint handling, loss plotting, and demo launch.

## How SimSiam works

Two augmented views of one image pass through the same encoder and projector. A predictor processes each projection. The loss is the average negative cosine similarity between one prediction and the detached representation from the other branch. This detached branch is stop-gradient and helps avoid representation collapse. After pretraining, the encoder is frozen and a linear classifier is trained with labels.

## Project structure

```text
models/       encoder, projector, predictor, SimSiam
datasets/     CIFAR-10 loader and augmentations
training/     pretraining, linear evaluation, supervised training
evaluation/   metrics, plots, test-set evaluation
app/          Gradio demo
utils/        seeds and checkpoint handling
configs/      YAML configuration
notebooks/    guided experiment notebook
```

## Troubleshooting

- `CIFAR-10 not found`: run `python main.py download` or check the folder layout.
- `Checkpoint not found`: run pretraining, then linear evaluation.
- Activation error: run `Set-ExecutionPolicy -Scope Process Bypass`.
- Slow CPU training: reduce `--limit` or `--epochs`.
- Gradio import error: activate `venv` and run `pip install -r requirements.txt`.

The project sets reproducible Python, NumPy, and PyTorch seeds through `utils/seed.py`. Small differences may occur across hardware and PyTorch versions.

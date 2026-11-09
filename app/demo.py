from pathlib import Path
import torch
from models.simsiam import SimSiam
from datasets.cifar10 import CLASSES
from utils.checkpoint import load_checkpoint

MODEL = None

def load_model(checkpoint='checkpoints/linear.pt'):
    global MODEL
    if not Path(checkpoint).exists(): return None
    encoder=SimSiam(); load_checkpoint('checkpoints/simsiam.pt', encoder); encoder.eval()
    classifier=torch.nn.Linear(512,10); state=torch.load(checkpoint,map_location='cpu',weights_only=False); classifier.load_state_dict(state['model']); classifier.eval(); MODEL=(encoder,classifier); return MODEL

def predict(image):
    import gradio as gr
    if MODEL is None: raise gr.Error('Checkpoint is missing. Run: python main.py train-linear')
    x=torch.tensor(image).permute(2,0,1).float()/255; x=torch.nn.functional.interpolate(x[None],size=(32,32))
    with torch.no_grad(): probs=torch.softmax(MODEL[1](MODEL[0].encode(x)),1)[0]
    return image, {CLASSES[i]:float(probs[i]) for i in range(10)}

def launch(checkpoint='checkpoints/linear.pt'):
    import gradio as gr
    load_model(checkpoint)
    gr.Interface(predict, gr.Image(type='numpy', label='Upload image'), [gr.Image(label='Uploaded image'), gr.Label(num_top_classes=10)], title='SimSiam CIFAR-10 Demo', description='Upload an image to classify it.').launch()

if __name__=='__main__': launch()

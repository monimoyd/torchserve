import urllib

import gradio as gr
import torch
import timm
import numpy as np

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from typing import Dict

import tarfile

MODEL_TAR_GZ_FILE= 'model.script.pt.tar.gz'
MODEL_FILE = 'model.script.pt'
  
# open file
with tarfile.open(MODEL_TAR_GZ_FILE) as f:  
    f.extractall('.')
  
model = torch.jit.load(MODEL_FILE)

# Download human-readable labels for Cifar10.
# get the classnames
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image: Image) -> Dict[str, float]:

    if image is None:
        return None

    image = image.resize((224, 224), Image.BILINEAR)
    image = np.array(image)
    image = torch.tensor(image[None, None, ...], dtype=torch.float32)
    image = image.squeeze(0)
    image = image.permute(0,3,1,2)

    preds = model.forward_jit(image)
    return {cifar10_labels[i]: float(preds[i]) for i in range(10)}

if __name__ == "__main__":
    gr.Interface(
        fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=10)
    ).launch(server_name="0.0.0.0")
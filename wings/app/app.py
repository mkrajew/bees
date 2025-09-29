import cv2
import gradio as gr
import numpy as np
import torch

from wings.config import MODELS_DIR
from wings.modeling.litnet import LitNet
from wings.modeling.loss import DiceLoss
from torchvision.io import decode_image
from wings.utils import load_image
from wings.visualizing.image_preprocess import unet_fit_rectangle_preprocess, final_coords
from wings.visualizing.visualize import visualize_coords

countries = ['AT', 'GR', 'HR', 'HU', 'MD', 'PL', 'RO', 'SI']

checkpoint_path = MODELS_DIR / 'unet-rectangle-epoch=08-val_loss=0.14-unet-training-rectangle_1.ckpt'
unet_model = torch.hub.load(
    'mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=False
)
num_epochs = 60
model = LitNet.load_from_checkpoint(checkpoint_path, model=unet_model, num_epochs=num_epochs, criterion=DiceLoss())
model.eval()


def ai(filepath):
    image_tensor, x_size, y_size = load_image(filepath, unet_fit_rectangle_preprocess)
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)

    output = model(image_tensor.cuda().unsqueeze(0))
    mask = torch.round(output).squeeze().detach().cpu().numpy()

    mask_coords = final_coords(mask, x_size, y_size)

    flat_coords = [coord for pair in mask_coords for coord in pair]  # Flatten list of tuples
    target_tensor = torch.tensor(flat_coords, dtype=torch.float32)

    # Visualize
    img = visualize_coords(img, target_tensor, spot_size=3, show=False)

    return img, mask_coords


demo = gr.Interface(
    fn=ai,
    inputs=gr.Image(type="filepath"),
    outputs=["image", gr.Textbox(lines=10)],
)

demo.launch()

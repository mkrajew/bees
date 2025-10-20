from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import time

from wings.config import MODELS_DIR, PROCESSED_DATA_DIR
from wings.gpa import recover_order
from wings.modeling.litnet import LitNet
from wings.modeling.loss import DiceLoss
from wings.utils import load_image
from wings.visualizing.image_preprocess import unet_fit_rectangle_preprocess, final_coords
from wings.visualizing.visualize import visualize_coords

section_labels = [str(i) for i in range(1, 20)]
red_label_colors = {
    "1": "#FF00FF",
    "2": "#FF00E6",
    "3": "#FF00CC",
    "4": "#FF00B3",
    "5": "#FF0099",
    "6": "#FF0080",
    "7": "#FF0066",
    "8": "#FF004D",
    "9": "#FF0033",
    "10": "#FF0019",
    "11": "#FF0000",
    "12": "#FF1A00",
    "13": "#FF3300",
    "14": "#FF4D00",
    "15": "#FF6600",
    "16": "#FF8000",
    "17": "#FF9900",
    "18": "#FFB300",
    "19": "#FFCC00"
}
green_label_colors = {
    "1": "#009933",
    "2": "#00A42D",
    "3": "#00AF27",
    "4": "#00BB22",
    "5": "#00C61C",
    "6": "#00D116",
    "7": "#00DD11",
    "8": "#00E80B",
    "9": "#00F305",
    "10": "#00FF00",
    "11": "#0BFF00",
    "12": "#16FF00",
    "13": "#22FF00",
    "14": "#2DFF00",
    "15": "#38FF00",
    "16": "#44FF00",
    "17": "#4FFF00",
    "18": "#5AFF00",
    "19": "#66FF00"
}

countries = ['AT', 'GR', 'HR', 'HU', 'MD', 'PL', 'RO', 'SI']
checkpoint_path = MODELS_DIR / 'unet-rectangle-epoch=08-val_loss=0.14-unet-training-rectangle_1.ckpt'
unet_model = torch.hub.load(
    'mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=False
)
num_epochs = 60
model = LitNet.load_from_checkpoint(checkpoint_path, model=unet_model, num_epochs=num_epochs, criterion=DiceLoss())
model.eval()

mean_coords = torch.load(
    PROCESSED_DATA_DIR / "mask_datasets" / 'rectangle' / "mean_shape.pth", weights_only=False
)


def update_submit_button_value(filepaths):
    if filepaths:
        filepaths = list(dict.fromkeys(filepaths))
        files_num = len(filepaths)
        if files_num == 1:
            return gr.update(value="Submit 1 file")
        elif files_num > 1:
            return gr.update(value=f"Submit {files_num} files")

    return gr.update(value=f"Submit")


def input_images(filepaths):
    return (
        gr.update(visible=False),  # entry_page
        filepaths,  # image_paths
    )


def sections(coords, img_sizes):
    sections_arr = []
    W, H = img_sizes
    Y, X = np.ogrid[:H, :W]
    r = 3
    R = 12

    for idx, (x, y) in enumerate(coords):
        y_img = H - y - 1
        mask_small = ((X - x) ** 2 + (Y - y_img) ** 2) < r ** 2
        mask_small = mask_small.astype(float)

        mask_large = ((X - x) ** 2 + (Y - y_img) ** 2) < R ** 2
        mask_large = mask_large.astype(float)
        mask_large[mask_small > 0] = 0
        mask_large *= 0.3

        combined_mask = mask_small + mask_large

        sections_arr.append((combined_mask, section_labels[idx]))

    return sections_arr


def update_output_image(filepaths, idx, coordinates, sizes):
    img_path = filepaths[idx]
    filename = Path(img_path).name
    img_size = sizes[idx]
    sections_arr = sections(coordinates[idx].cpu().numpy(), img_size)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = visualize_coords(img, coordinates[idx].flatten(), spot_size=2, show=False)

    return (
        gr.update(value=(img, sections_arr)),  # output_image
        gr.update(value=filename),  # filename_textbox
    )


def update_filename(new_filename, filepaths, idx):
    filepaths[idx] = new_filename
    return filepaths


def calculate_coords(filepaths, progress=gr.Progress(track_tqdm=True)):
    coordinates = []
    image_sizes = []
    for img_path in progress.tqdm(filepaths, desc="Processing images..."):
        image_tensor, x_size, y_size = load_image(img_path, unet_fit_rectangle_preprocess)
        image_sizes.append((x_size, y_size))

        output = model(image_tensor.cuda().unsqueeze(0))
        mask = torch.round(output).squeeze().detach().cpu().numpy()

        mask_coords = final_coords(mask, x_size, y_size)
        reordered = recover_order(mean_coords, torch.tensor(mask_coords))
        coordinates.append(reordered)
        time.sleep(1)

    return coordinates, image_sizes, gr.update(visible=True)  # image_page


def select_coordinate(evt: gr.SelectData):
    return evt.index  # selected_coordinate


def update_coordinates(coords, idx, sel_coord):
    if sel_coord:
        return (
            gr.update(value=f"## Point number {section_labels[sel_coord]}:"),  # point_description
            gr.update(value=int(coords[idx][sel_coord][0])),  # selected_section_x
            gr.update(value=int(coords[idx][sel_coord][1])),  # selected_section_y
            gr.update(interactive=True),  # edit_button
        )
    return (
        gr.update(), gr.update(), gr.update(), gr.update()
    )


def right_button_click(filepaths, idx):
    return (idx + 1) % len(filepaths)


def left_button_click(filepaths, idx):
    return (idx - 1) % len(filepaths)


def update_image_desc_md(filepaths, idx, sizes):
    return gr.update(value=f"# Image {idx + 1} / {len(filepaths)}\nSize: {sizes[idx][0]} x {sizes[idx][1]}")


with gr.Blocks() as demo:
    gr.Markdown("# Bee-wing coordinates marker test")
    with gr.Column() as entry_page:
        files_input = gr.File(
            file_types=['image'],
            file_count='multiple',
            label="Upload bee-wing images",
            height=500,
        )
        submit_button = gr.Button("Submit")


    image_paths = gr.State()
    image_idx = gr.State(0)
    image_coords = gr.State()
    images_sizes = gr.State()
    selected_coordinate = gr.State()

    with gr.Column(visible=False) as image_page:
        with gr.Column() as image_column:
            with gr.Row() as image_row:
                with gr.Column(scale=5) as image_slider:
                    output_image = gr.AnnotatedImage(color_map=green_label_colors, height=500, show_label=False)
                    with gr.Row(equal_height=True):
                        filename_scale = 10
                        left_button = gr.Button(value="<", size="lg", scale=2)
                        filename_textbox = gr.Textbox(
                            max_lines=1,
                            show_label=False,
                            scale=filename_scale,
                            interactive=True,
                            container=False,
                        )
                        right_button = gr.Button(value=">", size="lg", scale=2)
                with gr.Column(scale=1) as coordinates_data:
                    image_desc_md = gr.Markdown()
                    point_description = gr.Markdown(value="## Choose a point to see the coordinates")
                    selected_section_x = gr.Number(
                        label="X Coordinate:",
                        value=None,
                        placeholder="x",
                        scale=2,
                        interactive=False,
                        precision=0
                    )
                    selected_section_y = gr.Number(
                        label="Y Coordinate:",
                        value=None,
                        placeholder="y",
                        scale=2,
                        interactive=False,
                        precision=0
                    )
                    edit_button = gr.Button("Edit", interactive=False)
        with gr.Accordion(open=False, label="See all files") as files_list:
            gr.Markdown("# Test")

    files_input.change(
        fn=update_submit_button_value,
        inputs=files_input,
        outputs=submit_button,
    )

    submit_button.click(
        fn=input_images,
        inputs=files_input,
        outputs=[entry_page, image_paths],
    ).then(
        fn=calculate_coords,
        inputs=image_paths,
        outputs=[image_coords, images_sizes, image_page],
    ).then(
        fn=update_output_image,
        inputs=[image_paths, image_idx, image_coords, images_sizes],
        outputs=[output_image, filename_textbox],
    ).then(
        fn=update_image_desc_md,
        inputs=[image_paths, image_idx, images_sizes],
        outputs=image_desc_md
    )

    filename_textbox.submit(
        fn=update_filename,
        inputs=[filename_textbox, image_paths, image_idx],
        outputs=image_paths
    )

    output_image.select(
        fn=select_coordinate,
        outputs=selected_coordinate
    ).then(
        fn=update_coordinates,
        inputs=[image_coords, image_idx, selected_coordinate],
        outputs=[
            point_description,
            selected_section_x,
            selected_section_y,
            edit_button,
        ],
    )

    right_button.click(
        fn=right_button_click,
        inputs=[image_paths, image_idx],
        outputs=image_idx,
    ).then(
        fn=update_output_image,
        inputs=[image_paths, image_idx, image_coords, images_sizes],
        outputs=[output_image, filename_textbox],
    ).then(
        fn=update_coordinates,
        inputs=[image_coords, image_idx, selected_coordinate],
        outputs=[
            point_description,
            selected_section_x,
            selected_section_y,
            edit_button,
        ],
    ).then(
        fn=update_image_desc_md,
        inputs=[image_paths, image_idx, images_sizes],
        outputs=image_desc_md
    )

    left_button.click(
        fn=left_button_click,
        inputs=[image_paths, image_idx],
        outputs=image_idx,
    ).then(
        fn=update_output_image,
        inputs=[image_paths, image_idx, image_coords, images_sizes],
        outputs=[output_image, filename_textbox],
    ).then(
        fn=update_coordinates,
        inputs=[image_coords, image_idx, selected_coordinate],
        outputs=[
            point_description,
            selected_section_x,
            selected_section_y,
            edit_button,
        ],
    ).then(
        fn=update_image_desc_md,
        inputs=[image_paths, image_idx, images_sizes],
        outputs=image_desc_md
    )

if __name__ == '__main__':
    demo.launch()

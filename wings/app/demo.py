from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from wings.config import MODELS_DIR
from wings.modeling.litnet import LitNet
from wings.modeling.loss import DiceLoss
from wings.utils import load_image
from wings.visualizing.image_preprocess import unet_fit_rectangle_preprocess, final_coords

countries = ['AT', 'GR', 'HR', 'HU', 'MD', 'PL', 'RO', 'SI']
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

    output = model(image_tensor.cuda().unsqueeze(0))
    mask = torch.round(output).squeeze().detach().cpu().numpy()

    mask_coords = final_coords(mask, x_size, y_size)

    return mask_coords


# def ai(filepath):
#     img = cv2.imread(filepath)
#
#     height, width, _ = img.shape
#
#     num_points = 19
#     coordinates = []
#     for _ in range(num_points):
#         x = random.randint(0, width - 1)
#         y = random.randint(0, height - 1)
#         coordinates.append((x, y))
#
#     return coordinates


def show_image(img_path):
    img_name = Path(img_path).name
    sections_arr, coords = sections(img_path)
    df_arr = [(idx + 1, int(x), int(y)) for idx, (x, y) in enumerate(coords)]
    return (
        gr.update(visible=False),  # file_input
        coords,  # coords_state
        gr.update(value=img_name),  # filename
        gr.update(visible=True),  # filename_buttons_row
        gr.update(value=(img_path, sections_arr)),  # output_image
        gr.update(value=img_path),  # edit_image
        gr.update(visible=True),  # image_coords_row
        gr.update(visible=True),  # all_coords
        gr.update(value=df_arr),  # coords_df
    )


def sections(img_path):
    coords = ai(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sections_arr = []
    H, W = img.shape[:2]
    Y, X = np.ogrid[:H, :W]
    r = 3
    R = 9

    for idx, (x, y) in enumerate(coords):
        y_img = H - y
        mask_small = ((X - x) ** 2 + (Y - y_img) ** 2) < r ** 2
        mask_small = mask_small.astype(float)

        mask_large = ((X - x) ** 2 + (Y - y_img) ** 2) < R ** 2
        mask_large = mask_large.astype(float)
        mask_large[mask_small > 0] = 0
        mask_large *= 0.3

        combined_mask = mask_small + mask_large

        sections_arr.append((combined_mask, section_labels[idx]))

    return sections_arr, coords


with gr.Blocks() as demo:
    gr.Markdown("# Greetings to all the bees!")
    with gr.Tab("Single image") as single_image:
        coords_state = gr.State()
        chosen_annotation_state = gr.State()

        file_input = gr.File(file_types=['image'], label="Upload a bee-wing image", height=500)
        with gr.Row(visible=False, equal_height=True) as filename_buttons_row:
            filename = gr.Textbox(label="Verify file name:", interactive=True, scale=4)
            with gr.Column(scale=1):
                download_button = gr.Button(value="Download data")
                next_image_button = gr.Button(value="Next image")
        with gr.Row(visible=False) as image_coords_row:
            output_image = gr.AnnotatedImage(color_map=red_label_colors, scale=4, height=500)
            edit_image = gr.Image(visible=False, scale=4, height=500, interactive=True)
            with gr.Column(scale=1) as chosen:
                md_text = gr.Markdown(value="## Choose a point to see the coordinates")
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
                edit_button = gr.Button(value="Edit", scale=1, interactive=False)
                confirm_button = gr.Button(value="Confirm", visible=False)
                updated_coords_info = gr.Markdown(
                    value="Coordinates updated. (Not yes, does not work yet)",
                    visible=False
                )

        with gr.Accordion(visible=False, label="See all coordinates", open=False) as all_coords:
            coords_df = gr.Matrix(
                headers=["Point", "X", "Y"],
                datatype="number",
                column_widths=["10%", "20%", "20%"]
            )

        file_input.change(
            fn=show_image,
            inputs=file_input,
            outputs=[file_input, coords_state, filename, filename_buttons_row, output_image, edit_image,
                     image_coords_row, all_coords, coords_df]
        )


        def select_section(evt: gr.SelectData, coords):
            return (
                gr.update(value=f"## Point number {section_labels[evt.index]}:"),  # md_text
                gr.update(value=coords[evt.index][0]),  # selected_section_x
                gr.update(value=coords[evt.index][1]),  # selected_section_y
                gr.update(interactive=True),  # edit_button
                gr.update(visible=False),  # updated_coords_info
                evt.index,  # chosen_annotation_state
            )


        output_image.select(
            select_section,
            inputs=[coords_state],
            outputs=[
                md_text,
                selected_section_x,
                selected_section_y,
                edit_button,
                updated_coords_info,
                chosen_annotation_state
            ]
        )
        edit_button.click(
            fn=lambda: (
                gr.update(visible=False),  # edit_button
                gr.update(visible=True),  # confirm_button
                gr.update(visible=False),  # output_image
                gr.update(visible=True),  # edit_image
                gr.update(value="## Click on a new point in the image or type in new values in the boxes"),  # md_text
                gr.update(interactive=True),  # selected_section_x
                gr.update(interactive=True),  # selected_section_y
            ),
            inputs=None,
            outputs=[edit_button, confirm_button, output_image, edit_image, md_text, selected_section_x,
                     selected_section_y]
        )


        def get_click_coordinates(img, evt: gr.SelectData):
            return gr.update(value=evt.index[0]), gr.update(value=evt.index[1])


        edit_image.select(get_click_coordinates, inputs=edit_image, outputs=[selected_section_x, selected_section_y])


        def click_confirm_button(coords, x_coord, y_coord, point):
            coords[point] = x_coord, y_coord
            return (
                gr.update(visible=True),  # edit_button
                gr.update(visible=False),  # confirm_button
                gr.update(visible=True),  # output_image
                gr.update(visible=False),  # edit_image
                gr.update(value="## Choose a point to see the coordinates"),  # md_text
                gr.update(interactive=True),  # selected_section_x
                gr.update(interactive=True),  # selected_section_y
                gr.update(visible=True),  # updated_coords_info
                coords, # coords_state
            )

        confirm_button.click(
            fn=click_confirm_button,
            inputs=[coords_state, selected_section_x, selected_section_y, chosen_annotation_state],
            outputs=[edit_button, confirm_button, output_image, edit_image, md_text, selected_section_x,
                     selected_section_y, updated_coords_info, coords_state]
        )

        download_button.click(
            fn=lambda: gr.update(value="This option does not work yet"),
            inputs=None,
            outputs=download_button
        )


        def reset_ui():
            return (
                gr.update(value=None, visible=True),  # file_input visible again
                None,  # coords_state cleared
                gr.update(visible=False),  # filename_buttons_row hidden
                gr.update(visible=False),  # image_coords_row hidden
                gr.update(visible=False),  # all_coords hidden
                gr.update(value="### Choose a point to see the coordinates"),  # md_text
                gr.update(value=None, interactive=False),  # selected_section_x
                gr.update(value=None, interactive=False),  # selected_section_y
                gr.update(interactive=False),  # edit_button
            )


        next_image_button.click(
            fn=reset_ui,
            inputs=None,
            outputs=[
                file_input,
                coords_state,
                filename_buttons_row,
                image_coords_row,
                all_coords,
                md_text,
                selected_section_x,
                selected_section_y,
                edit_button,
            ]
        )

    with gr.Tab("Many images") as many_images:
        gr.Markdown("# Does not work yet")

if __name__ == "__main__":
    demo.launch()

import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

from wings.app.images import WingImage
from wings.config import MODELS_DIR, PROCESSED_DATA_DIR
from wings.modeling.litnet import LitNet
from wings.modeling.loss import DiceLoss
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
green_label_colors_orig = {
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
green_label_colors = green_label_colors_orig.copy()
red_color_str = "#FF0000"

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


def input_images(filepaths, progress=gr.Progress(track_tqdm=True)):
    images = []
    check_idxs = []
    for idx, filepath in enumerate(progress.tqdm(filepaths, desc="Processing images...")):
        image = WingImage(filepath, model, mean_coords, section_labels)
        images.append(image)
        if image.check_carefully:
            check_idxs.append(idx)

    return images, check_idxs


def add_images(new_filepaths, images, check_idxs, progress=gr.Progress(track_tqdm=True)):
    for idx, filepath in enumerate(progress.tqdm(new_filepaths, desc="Processing images...")):
        image = WingImage(filepath, model, mean_coords, section_labels)
        if image not in images:
            images.append(image)
            if image.check_carefully:
                check_idxs.append(idx)

    return images, check_idxs


def update_output_image(images, idx):
    image = images[idx]
    img = image.image
    img = visualize_coords(img, image.coordinates.flatten(), spot_size=2, show=False)

    return (
        gr.update(value=(img, image.sections), color_map=green_label_colors),  # output_image
        gr.update(value=image.filename),  # filename_textbox
    )


def update_image_desc_md(images, idx):
    sizes = images[idx].size
    return gr.update(value=f"# Image {idx + 1} / {len(images)}\nSize: {sizes[0]} x {sizes[1]}")


def calc_dataframe(images):
    columns = ["file"]
    for i in range(1, 20):
        columns.append(f"x{i}")
        columns.append(f"y{i}")

    rows = []
    for image in images:
        flattened_coords = image.coordinates.flatten()
        flattened_coords = [int(i) for i in flattened_coords]
        row = [image.filename] + flattened_coords
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def update_dataframe(images):
    wings_df = calc_dataframe(images)
    return gr.update(value=wings_df)


def update_filename(new_filename, images, idx):
    images[idx].filename = new_filename
    return gr.update(value=images[idx].filename), images


def select_coordinate(evt: gr.SelectData):
    global green_label_colors
    green_label_colors = green_label_colors_orig.copy()
    green_label_colors[f"{evt.index + 1}"] = red_color_str
    return evt.index  # selected_coordinate


def update_coordinates(images, idx, sel_coord):
    if sel_coord is not None:
        return (
            gr.update(value=f"## Point number {section_labels[sel_coord]}:"),  # point_description
            gr.update(value=int(images[idx].coordinates[sel_coord][0])),  # selected_section_x
            gr.update(value=int(images[idx].coordinates[sel_coord][1])),  # selected_section_y
            gr.update(interactive=True),  # edit_button
        )
    return (
        gr.update(), gr.update(), gr.update(), gr.update()
    )


def right_button_click(filepaths, idx):
    return (idx + 1) % len(filepaths)


def left_button_click(filepaths, idx):
    return (idx - 1) % len(filepaths)


def generate_data(options, images, user_tmp, progress=gr.Progress(track_tqdm=True)):
    if user_tmp is None:
        user_tmp = tempfile.TemporaryDirectory()
    tmpdir = Path(user_tmp.name)

    if options == "CSV":
        file_name = "landmarks.csv"
        filepath = tmpdir / file_name
        df_coords = calc_dataframe(images)
        df_coords.to_csv(filepath, index=False)
    else:
        file_name = "images.zip" if options == "Images" else "landmarks.zip"
        filepath = tmpdir / file_name
        with zipfile.ZipFile(filepath, "w") as zip_imgs:
            for image in progress.tqdm(images):
                img_path = image.generate_image_with_meta_landmarks()
                zip_imgs.write(img_path, arcname=img_path.name)
            if options == "Both":
                csv_path = "landmarks.csv"
                df_coords = calc_dataframe(images)
                df_coords.to_csv(tmpdir / csv_path, index=False)
                zip_imgs.write(tmpdir / csv_path, arcname=csv_path)
    return gr.update(value=filepath, interactive=True), user_tmp


def clean_temp(user_tmp):
    if user_tmp is not None:
        user_tmp.cleanup()

    return None


with gr.Blocks() as demo:
    gr.Markdown("# WingAI")
    gr.Markdown("Automated Landmark Detection for Bee Wing Morphometrics")

    with gr.Column() as entry_page:
        files_input = gr.File(
            file_types=['.png'],
            file_count='multiple',
            label="Upload bee-wing images",
            height=500,
        )
        submit_button = gr.Button("Submit")

    wing_images = gr.State()
    image_idx = gr.State(0)
    image_check_idxs = gr.State()
    selected_coordinate = gr.State(None)
    temp_dir = gr.State(None)

    with gr.Column(visible=False) as image_page:
        with gr.Row(equal_height=True):
            generate_data_button = gr.Button("Generate Data")
            download_type = gr.Radio(
                choices=["CSV", "Images", "Both"],
                value="CSV",
                label="Choose where to save landmarks",
                container=False,
                show_label=False,
                interactive=True
            )
            download_button = gr.DownloadButton(label="Download Data", interactive=False, variant='primary')
        with gr.Row() as image_row:
            with gr.Column(scale=5) as image_viewer:
                output_image = gr.AnnotatedImage(
                    color_map=green_label_colors,
                    height=500,
                    show_label=False,
                    show_fullscreen_button=False
                )
                edit_image = gr.Image(
                    visible=False,
                    interactive=False,
                    show_download_button=False,
                    show_fullscreen_button=False
                )
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
                add_images_button = gr.UploadButton(
                    label="Add Images",
                    file_types=['image'],
                    file_count='multiple',
                )
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
                with gr.Row(visible=False) as confirm_cancel_buttons:
                    confirm_button = gr.Button(value="Confirm", interactive=True)
                    cancel_button = gr.Button(value="Cancel", interactive=True)
        with gr.Accordion(open=False, label="See all files") as files_list:
            df = gr.Dataframe(show_row_numbers=True)

    files_input.change(
        fn=update_submit_button_value,
        inputs=files_input,
        outputs=submit_button,
    )

    submit_button.click(
        fn=lambda: (
            gr.update(visible=False),
            gr.update(visible=True)
        ),
        outputs=[entry_page, image_page]
    ).then(
        fn=input_images,
        inputs=files_input,
        outputs=[wing_images, image_check_idxs],
        show_progress_on=output_image,
    ).success(
        fn=update_output_image,
        inputs=[wing_images, image_idx],
        outputs=[output_image, filename_textbox],
    ).then(
        fn=update_image_desc_md,
        inputs=[wing_images, image_idx],
        outputs=image_desc_md
    ).then(
        fn=update_dataframe,
        inputs=wing_images,
        outputs=df,
    )

    filename_textbox.submit(
        fn=update_filename,
        inputs=[filename_textbox, wing_images, image_idx],
        outputs=[filename_textbox, wing_images]
    ).then(
        fn=update_dataframe,
        inputs=wing_images,
        outputs=df,
    )

    output_image.select(
        fn=select_coordinate,
        outputs=selected_coordinate
    ).then(
        fn=update_coordinates,
        inputs=[wing_images, image_idx, selected_coordinate],
        outputs=[
            point_description,
            selected_section_x,
            selected_section_y,
            edit_button,
        ],
    ).then(
        fn=update_output_image,
        inputs=[wing_images, image_idx],
        outputs=[output_image, filename_textbox],
    )

    right_button.click(
        fn=right_button_click,
        inputs=[wing_images, image_idx],
        outputs=image_idx,
    ).then(
        fn=update_output_image,
        inputs=[wing_images, image_idx],
        outputs=[output_image, filename_textbox],
    ).then(
        fn=update_coordinates,
        inputs=[wing_images, image_idx, selected_coordinate],
        outputs=[
            point_description,
            selected_section_x,
            selected_section_y,
            edit_button,
        ],
    ).then(
        fn=update_image_desc_md,
        inputs=[wing_images, image_idx],
        outputs=image_desc_md
    )

    left_button.click(
        fn=left_button_click,
        inputs=[wing_images, image_idx],
        outputs=image_idx,
    ).then(
        fn=update_output_image,
        inputs=[wing_images, image_idx],
        outputs=[output_image, filename_textbox],
    ).then(
        fn=update_coordinates,
        inputs=[wing_images, image_idx, selected_coordinate],
        outputs=[
            point_description,
            selected_section_x,
            selected_section_y,
            edit_button,
        ],
    ).then(
        fn=update_image_desc_md,
        inputs=[wing_images, image_idx],
        outputs=image_desc_md
    )

    add_images_button.upload(
        fn=add_images,
        inputs=[add_images_button, wing_images, image_check_idxs],
        outputs=[wing_images, image_check_idxs],
        show_progress_on=output_image,
    ).then(
        fn=update_image_desc_md,
        inputs=[wing_images, image_idx],
        outputs=image_desc_md
    ).then(
        fn=update_dataframe,
        inputs=wing_images,
        outputs=df,
    )

    generate_data_button.click(
        fn=generate_data,
        inputs=[download_type, wing_images, temp_dir],
        outputs=[download_button, temp_dir],
    )

    download_button.click(
        fn=clean_temp,
        inputs=temp_dir,
        outputs=temp_dir,
    )

if __name__ == '__main__':
    demo.launch()

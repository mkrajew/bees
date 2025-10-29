import tempfile
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, PngImagePlugin

from wings.config import MODELS_DIR, PROCESSED_DATA_DIR
from wings.gpa import handle_coordinates
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


def input_images(filepaths):
    return (
        gr.update(visible=False),  # entry_page
        filepaths,  # image_paths
        gr.update(visible=True),  # image_page
    )


def add_images(new_filepaths, old_filepaths, sizes, coords):
    new_filepaths = [filepath for filepath in new_filepaths if filepath not in old_filepaths]
    new_coords, new_sizes = calculate_coords(new_filepaths)
    old_filepaths.extend(new_filepaths)
    sizes.extend(new_sizes)
    coords.extend(new_coords)

    return old_filepaths, sizes, coords


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
        gr.update(value=(img, sections_arr), color_map=green_label_colors),  # output_image
        gr.update(value=filename),  # filename_textbox
    )


def update_filename(new_filename, filepaths, idx):
    filepaths[idx] = new_filename
    # TODO: to zmienia cala sciezke, niech zmieni sie tylko nazwa
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
        mask_coords = torch.tensor(mask_coords)
        reordered = handle_coordinates(mask_coords, mean_coords)
        coordinates.append(reordered)

    return coordinates, image_sizes


def select_coordinate(evt: gr.SelectData):
    global green_label_colors
    green_label_colors = green_label_colors_orig.copy()
    green_label_colors[f"{evt.index + 1}"] = red_color_str
    return evt.index  # selected_coordinate


def update_coordinates(coords, idx, sel_coord):
    if sel_coord is not None:
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


def update_dataframe(filepaths, coords):
    columns = ["file"]
    for i in range(1, 20):
        columns.append(f"x{i}")
        columns.append(f"y{i}")

    rows = []

    for path, coord_set in zip(filepaths, coords):
        filename = Path(path).name
        flattened_coords = []
        for x, y in coord_set:
            flattened_coords.extend([int(x), int(y)])

        row = [filename] + flattened_coords
        rows.append(row)

    df_coords = pd.DataFrame(rows, columns=columns)

    return df_coords


def generate_data(options, filepaths, coords, df_coords, sizes, user_tmp):
    if user_tmp is None:
        user_tmp = tempfile.TemporaryDirectory()
    tmpdir = Path(user_tmp.name)

    if options == "CSV":
        file_name = "landmarks.csv"
        filepath = tmpdir / file_name
        df_coords.to_csv(filepath, index=False)
    else:
        file_name = "images.zip" if options == "Images" else "landmarks.zip"
        filepath = tmpdir / file_name
        with zipfile.ZipFile(filepath, "w") as zip_imgs:
            for img, coords, sizes in zip(filepaths, coords, sizes):
                img_path = Path(img.name)  # Gradio File object
                img = Image.open(img_path)
                img.load()

                y_size = sizes[1]
                coords_np = coords.detach().cpu().numpy().copy()
                coords_np[:, 1] = y_size - coords_np[:, 1] - 1
                labels_str = " ".join(str(int(x)) for x in coords_np.flatten())
                meta_str = f"landmarks:{labels_str};"
                png_info = PngImagePlugin.PngInfo()
                png_info.add_text("IdentiFly", meta_str)

                # TODO: check path if it has .png extension
                img.save(img_path, pnginfo=png_info)
                zip_imgs.write(img_path, arcname=img_path.name)
            if options == "Both":
                csv_path = "landmarks.csv"
                df_coords.to_csv(csv_path, index=False)
                zip_imgs.write(csv_path)
    return gr.update(value=filepath, interactive=True), user_tmp


def show_edit_image(paths, idx, coords, edit_coord_idx):
    img_path = paths[idx]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    other_coords = torch.cat([coords[idx][:edit_coord_idx], coords[idx][edit_coord_idx + 1:]], dim=0)
    img = visualize_coords(img, other_coords.flatten(), spot_size=2, show=False)
    img = visualize_coords(img, coords[idx][edit_coord_idx].flatten(), spot_size=2, color=(255, 0, 0), show=False)

    return (
        gr.update(visible=False),  # output_image
        gr.update(value=img, visible=True),  # edit_image
        gr.update(interactive=False),  # left_button
        gr.update(interactive=False),  # right_button
        gr.update(visible=True),  # confirm_cancel_buttons
        gr.update(visible=False),  # edit_button
    )


def get_edit_coordinates(img, evt: gr.SelectData):
    x = evt.index[0]
    y = img.shape[0] - 1 - evt.index[1]
    return x, y


def show_edit_point(paths, idx, coords, edit_coord_idx, tmp_coords):
    # TODO: correct the repetition
    img_path = paths[idx]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    other_coords = torch.cat([coords[idx][:edit_coord_idx], coords[idx][edit_coord_idx + 1:]], dim=0)
    img = visualize_coords(img, other_coords.flatten(), spot_size=2, show=False)
    img = visualize_coords(img, torch.tensor(tmp_coords), spot_size=2, color=(255, 0, 0), show=False)

    return (
        gr.update(value=img),
        gr.update(value=tmp_coords[0]),  # selected_section_x
        gr.update(value=tmp_coords[1]),  # selected_section_y
    )


def cancel_button_click():
    return (
        gr.update(visible=False),  # confirm_cancel_buttons
        gr.update(visible=True),  # edit_button,
        gr.update(visible=False),  # edit_image,
        gr.update(visible=True),  # output_image,
        gr.update(interactive=True),  # left_button
        gr.update(interactive=True),  # right_button
    )


def confirm_edit_coords(coords, idx, sel_coord_idx, temp):
    coords[idx][sel_coord_idx] = torch.tensor(temp)
    return coords, None


def clean_temp(user_tmp):
    if user_tmp is not None:
        user_tmp.cleanup()

    return None


with (gr.Blocks() as demo):
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

    image_paths = gr.State()
    image_idx = gr.State(0)
    image_coords = gr.State()
    images_sizes = gr.State()
    selected_coordinate = gr.State(None)
    all_coords_df = gr.State(pd.DataFrame())
    tmp_edit_coords = gr.State()
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
                with gr.Row():
                    image_desc_md = gr.Markdown()
                    add_images_button = gr.UploadButton(
                        label="Add Images",
                        file_types=['image'],
                        file_count='multiple'
                    )
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
        fn=input_images,
        inputs=files_input,
        outputs=[entry_page, image_paths, image_page],
    ).then(
        fn=calculate_coords,
        inputs=image_paths,
        outputs=[image_coords, images_sizes],
        show_progress_on=output_image,
    ).success(
        fn=update_output_image,
        inputs=[image_paths, image_idx, image_coords, images_sizes],
        outputs=[output_image, filename_textbox],
    ).then(
        fn=update_image_desc_md,
        inputs=[image_paths, image_idx, images_sizes],
        outputs=image_desc_md
    ).then(
        fn=update_dataframe,
        inputs=[image_paths, image_coords],
        outputs=all_coords_df,
    ).then(
        fn=lambda coords: gr.update(value=coords),
        inputs=all_coords_df,
        outputs=df,
    )

    filename_textbox.submit(
        fn=update_filename,
        inputs=[filename_textbox, image_paths, image_idx],
        outputs=image_paths
    ).then(
        fn=update_dataframe,
        inputs=[image_paths, image_coords],
        outputs=all_coords_df,
    ).then(
        fn=lambda coords: gr.update(value=coords),
        inputs=all_coords_df,
        outputs=df,
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
    ).then(
        fn=update_output_image,
        inputs=[image_paths, image_idx, image_coords, images_sizes],
        outputs=[output_image, filename_textbox],
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

    generate_data_button.click(
        fn=generate_data,
        inputs=[download_type, image_paths, image_coords, all_coords_df, images_sizes, temp_dir],
        outputs=[download_button, temp_dir],
    )

    download_button.click(
        fn=clean_temp,
        inputs=temp_dir,
        outputs=temp_dir,
    )

    add_images_button.upload(
        fn=add_images,
        inputs=[add_images_button, image_paths, images_sizes, image_coords],
        outputs=[image_paths, images_sizes, image_coords],
        show_progress_on=output_image,
    ).then(
        fn=update_image_desc_md,
        inputs=[image_paths, image_idx, images_sizes],
        outputs=image_desc_md
    ).then(
        fn=update_dataframe,
        inputs=[image_paths, image_coords],
        outputs=all_coords_df,
    ).then(
        fn=lambda coords: gr.update(value=coords),
        inputs=all_coords_df,
        outputs=df,
    )

    edit_button.click(
        fn=show_edit_image,
        inputs=[image_paths, image_idx, image_coords, selected_coordinate],
        outputs=[output_image, edit_image, left_button, right_button, confirm_cancel_buttons, edit_button]
    )

    edit_image.select(
        fn=get_edit_coordinates,
        inputs=[edit_image],
        outputs=[tmp_edit_coords]
    ).then(
        fn=show_edit_point,
        inputs=[image_paths, image_idx, image_coords, selected_coordinate, tmp_edit_coords],
        outputs=[edit_image, selected_section_x, selected_section_y]
    )

    confirm_button.click(
        fn=confirm_edit_coords,
        inputs=[image_coords, image_idx, selected_coordinate, tmp_edit_coords],
        outputs=[image_coords, tmp_edit_coords],
    ).then(
        fn=cancel_button_click,
        outputs=[confirm_cancel_buttons, edit_button, edit_image, output_image, left_button, right_button]
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
        fn=update_dataframe,
        inputs=[image_paths, image_coords],
        outputs=all_coords_df,
    ).then(
        fn=lambda coords: gr.update(value=coords),
        inputs=all_coords_df,
        outputs=df,
    )

    cancel_button.click(
        fn=cancel_button_click,
        outputs=[confirm_cancel_buttons, edit_button, edit_image, output_image, left_button, right_button]
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

    if __name__ == '__main__':
        demo.launch()

import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd

from wings.app import model, mean_coords, section_labels, green_label_colors_orig, red_color_str
from wings.app.images import WingImage
from wings.visualizing.visualize import visualize_coords

green_label_colors = green_label_colors_orig.copy()


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
    return None, gr.update(interactive=False)

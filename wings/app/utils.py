import random
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd
import torch

from wings.app import model, mean_coords, section_labels, green_label_colors_orig, red_color_str
from wings.app.images import WingImage, LoadImageError
from wings.config import APP_DIR
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
        try:
            image = WingImage(filepath, model, mean_coords, section_labels)
            images.append(image)
        except LoadImageError as e:
            gr.Warning(str(e))

    if len(images) == 0:
        raise gr.Error(message="No correct images", print_exception=False)

    for idx, image in enumerate(images):
        if image.check_carefully:
            check_idxs.append(idx)

    return images, check_idxs


def add_images(new_filepaths, images, check_idxs, progress=gr.Progress(track_tqdm=True)):
    old_images_num = len(images)
    for idx, filepath in enumerate(progress.tqdm(new_filepaths, desc="Processing images...")):
        try:
            image = WingImage(filepath, model, mean_coords, section_labels)
            if image not in images:
                images.append(image)
        except LoadImageError as e:
            gr.Warning(str(e))

    if len(images) == 0:
        raise gr.Error(message="No correct images", print_exception=False)

    for idx, image in enumerate(images[old_images_num:], start=old_images_num):
        if image.check_carefully:
            check_idxs.append(idx)

    return images, check_idxs


def input_images_failure():
    image_path = APP_DIR / "bee.png"
    width, height = 640, 480
    box_w = random.randint(100, 300)
    box_h = random.randint(100, 250)
    x1 = random.randint(0, width - box_w)
    y1 = random.randint(0, height - box_h)
    x2 = x1 + box_w
    y2 = y1 + box_h
    bbox = (x1, y1, x2, y2)
    return gr.update(value=(str(image_path), [(bbox, "1")]))


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


def right_button_click(filepaths, idx, check_images):
    if idx in check_images:
        check_images.remove(idx)
    return (idx + 1) % len(filepaths), check_images


def left_button_click(filepaths, idx, check_images):
    if idx in check_images:
        check_images.remove(idx)
    return (idx - 1) % len(filepaths), check_images


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


def show_edit_image(images, idx, edit_coord_idx):
    img = images[idx].image
    coords = images[idx].coordinates
    other_coords = torch.cat([coords[:edit_coord_idx], coords[edit_coord_idx + 1:]], dim=0)
    img = visualize_coords(img, other_coords.flatten(), spot_size=2, show=False)
    img = visualize_coords(img, coords[edit_coord_idx].flatten(), spot_size=2, color=(255, 0, 0), show=False)

    return (
        gr.update(visible=False),  # output_image
        gr.update(value=img, visible=True),  # edit_image
        gr.update(interactive=False),  # left_button
        gr.update(interactive=False),  # right_button
        gr.update(visible=True),  # confirm_cancel_buttons
        gr.update(visible=False),  # edit_button
        gr.update(interactive=False),  # add_images_button
        gr.update(interactive=False),  # generate_button
    )


def cancel_button_click():
    return (
        gr.update(visible=False),  # confirm_cancel_buttons
        gr.update(visible=True),  # edit_button,
        gr.update(visible=False),  # edit_image,
        gr.update(visible=True),  # output_image,
        gr.update(interactive=True),  # left_button
        gr.update(interactive=True),  # right_button
        gr.update(interactive=True),  # add_images_button
        gr.update(interactive=True),  # generate_button
    )


def get_edit_coordinates(img, evt: gr.SelectData):
    x = evt.index[0]
    y = img.shape[0] - 1 - evt.index[1]
    return x, y


def confirm_edit_coords(images, idx, sel_coord_idx, temp):
    coords = images[idx].coordinates
    coords[sel_coord_idx] = torch.tensor(temp)
    images[idx].coordinates = coords
    return images, None


def show_edit_point(images, idx, edit_coord_idx, tmp_coords):
    img = images[idx].image
    coords = images[idx].coordinates
    other_coords = torch.cat([coords[:edit_coord_idx], coords[edit_coord_idx + 1:]], dim=0)
    img = visualize_coords(img, other_coords.flatten(), spot_size=2, show=False)
    img = visualize_coords(img, torch.tensor(tmp_coords), spot_size=2, color=(255, 0, 0), show=False)

    return (
        gr.update(value=img),
        gr.update(value=tmp_coords[0]),  # selected_section_x
        gr.update(value=tmp_coords[1]),  # selected_section_y
    )


def show_check_images(check_ids):
    text = generate_check_images_text(check_ids)
    return gr.update(value=text), gr.update(visible=len(check_ids) > 0)


def generate_check_images_text(check_ids):
    text = ""
    if len(check_ids) == 1:
        text = f"### Please carefully check image {check_ids[0] + 1}."
    elif len(check_ids) >= 1:
        text = "### Please carefully check images "
        for check_id in check_ids[:-2]:
            text += f"{check_id + 1}, "
        text += f"{check_ids[-2] + 1} and {check_ids[-1] + 1}."

    return text


def next_check_image(check_images, idx, images):
    if idx in check_images:
        check_images.remove(idx)
    if len(check_images) > 0:
        idx = check_images[0]
    else:
        idx = (idx + 1) % len(images)

    return check_images, idx


def reset_app():
    global green_label_colors
    green_label_colors = green_label_colors_orig.copy()
    return (
        gr.update(visible=True),   # show entry page
        gr.update(visible=False),  # hide image page
        gr.update(value=None),  # file_input
        gr.update(value="Submit"),  # submit_button
        None, 0, None, None, None, None,  # reset all states
        gr.update(value=None),  # selected_section_x
        gr.update(value=None),  # selected_section_y
        gr.update(value="## Choose a point to see the coordinates") # point_description
    )

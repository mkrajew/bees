import random

import cv2
import gradio as gr
import numpy as np

section_labels = [str(i) for i in range(1, 20)]
label_colors = {
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


def ai(filepath):
    img = cv2.imread(filepath)

    height, width, _ = img.shape

    num_points = 19
    coordinates = []
    for _ in range(num_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        coordinates.append((x, y))

    return coordinates


def show_image(img_path):
    img_name = img_path.split("/")[-1]
    sections_arr, coords = sections(img_path)
    return (
        gr.update(visible=False),
        coords,
        gr.update(value=img_name, visible=True),
        gr.update(value=(img_path, sections_arr)),
        gr.update(visible=True)
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
    gr.Markdown("# Greeting for all the bees!")
    coords_state = gr.State()

    file_input = gr.File(file_types=['image'], label="Upload a bee-wing image")
    filename = gr.Textbox(visible=False, label="Filename:")
    with gr.Row(visible=False) as image_coords_row:
        output_image = gr.AnnotatedImage(color_map=label_colors, scale=4, height=500)
        with gr.Column(scale=1) as chosen:
            md_text = gr.Markdown(value="### Choose a point to see the coordinates")
            selected_section_x = gr.Textbox(label="X Coordinate:", max_lines=1, scale=2)
            selected_section_y = gr.Textbox(label="Y Coordinate:", max_lines=1, scale=2)
            edit_button = gr.Button(value="Edit", scale=1)

    file_input.change(fn=show_image, inputs=file_input, outputs=[file_input, coords_state, filename, output_image, image_coords_row])


    def select_section(evt: gr.SelectData, coords):
        return (
            gr.update(value=f"## Point number {section_labels[evt.index]}:"),
            gr.update(value=coords[evt.index][0]),
            gr.update(value=coords[evt.index][1])
        )


    output_image.select(select_section, [coords_state], [md_text, selected_section_x, selected_section_y])

if __name__ == "__main__":
    demo.launch()

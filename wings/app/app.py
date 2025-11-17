from utils import *

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
    tmp_edit_coords = gr.State()

    with gr.Column(visible=False) as image_page:
        with gr.Row() as image_row:
            with gr.Column(scale=5) as image_viewer:
                with gr.Row(equal_height=True, visible=False) as check_images_row:
                    check_images_info_md = gr.Markdown()
                    next_check_button = gr.Button(value="Next image", size="sm", variant="primary")
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
        with gr.Accordion(open=False, label="Download data"):
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
        with gr.Accordion(open=False, label="See all files") as files_list:
            df = gr.Dataframe(show_row_numbers=True)
        reset_button = gr.Button(value="Reset", size="sm")

    files_input.change(
        fn=update_submit_button_value,
        inputs=files_input,
        outputs=submit_button,
    )

    submit_button_click = submit_button.click(
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
    )
    submit_button_click.failure(
        fn=input_images_failure,
        outputs=output_image,
    )
    submit_button_click.success(
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
    ).then(
        fn=show_check_images,
        inputs=image_check_idxs,
        outputs=[check_images_info_md, check_images_row],
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
        inputs=[wing_images, image_idx, image_check_idxs],
        outputs=[image_idx, image_check_idxs],
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
    ).then(
        fn=show_check_images,
        inputs=image_check_idxs,
        outputs=[check_images_info_md, check_images_row],
    )

    left_button.click(
        fn=left_button_click,
        inputs=[wing_images, image_idx, image_check_idxs],
        outputs=[image_idx, image_check_idxs],
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
    ).then(
        fn=show_check_images,
        inputs=image_check_idxs,
        outputs=[check_images_info_md, check_images_row],
    )

    add_images_upload = add_images_button.upload(
        fn=add_images,
        inputs=[add_images_button, wing_images, image_check_idxs],
        outputs=[wing_images, image_check_idxs],
        show_progress_on=output_image,
    )

    add_images_upload.failure(
        fn=input_images_failure,
        outputs=output_image,
    )

    add_images_upload.success(
        fn=update_image_desc_md,
        inputs=[wing_images, image_idx],
        outputs=image_desc_md
    ).then(
        fn=update_dataframe,
        inputs=wing_images,
        outputs=df,
    ).then(
        fn=show_check_images,
        inputs=image_check_idxs,
        outputs=[check_images_info_md, check_images_row],
    )

    generate_data_button.click(
        fn=generate_data,
        inputs=[download_type, wing_images, temp_dir],
        outputs=[download_button, temp_dir],
        show_progress_on=download_type,
    )

    download_button.click(
        fn=clean_temp,
        inputs=temp_dir,
        outputs=[temp_dir, download_button],
    )

    edit_button.click(
        fn=show_edit_image,
        inputs=[wing_images, image_idx, selected_coordinate],
        outputs=[
            output_image,
            edit_image,
            left_button,
            right_button,
            confirm_cancel_buttons,
            edit_button,
            add_images_button,
            generate_data_button,
        ]
    )

    edit_image.select(
        fn=get_edit_coordinates,
        inputs=[edit_image],
        outputs=[tmp_edit_coords]
    ).then(
        fn=show_edit_point,
        inputs=[wing_images, image_idx, selected_coordinate, tmp_edit_coords],
        outputs=[edit_image, selected_section_x, selected_section_y]
    )

    confirm_button.click(
        fn=confirm_edit_coords,
        inputs=[wing_images, image_idx, selected_coordinate, tmp_edit_coords],
        outputs=[wing_images, tmp_edit_coords],
    ).then(
        fn=cancel_button_click,
        outputs=[
            confirm_cancel_buttons,
            edit_button,
            edit_image,
            output_image,
            left_button,
            right_button,
            add_images_button,
            generate_data_button,
        ]
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
        fn=update_dataframe,
        inputs=wing_images,
        outputs=df,
    )

    cancel_button.click(
        fn=cancel_button_click,
        outputs=[
            confirm_cancel_buttons,
            edit_button,
            edit_image,
            output_image,
            left_button,
            right_button,
            add_images_button,
            generate_data_button,
        ]
    ).then(
        fn=update_coordinates,
        inputs=[wing_images, image_idx, selected_coordinate],
        outputs=[
            point_description,
            selected_section_x,
            selected_section_y,
            edit_button,
        ],
    )

    next_check_button.click(
        fn=next_check_image,
        inputs=[image_check_idxs, image_idx, wing_images],
        outputs=[image_check_idxs, image_idx],
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
    ).then(
        fn=show_check_images,
        inputs=image_check_idxs,
        outputs=[check_images_info_md, check_images_row],
    )

    reset_button.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            entry_page,
            image_page,
            files_input,
            submit_button,
            wing_images,
            image_idx,
            image_check_idxs,
            selected_coordinate,
            temp_dir,
            tmp_edit_coords,
            selected_section_x,
            selected_section_y,
            point_description,
        ]
    )

if __name__ == '__main__':
    demo.launch()

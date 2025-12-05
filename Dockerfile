FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install gradio==5.47.0 lightning loguru opencv-python scipy scikit-learn matplotlib

WORKDIR /app
COPY ./wings /app/wings
COPY ./data/processed/mask_datasets/rectangle/mean_shape.pth /app/data/processed/mask_datasets/rectangle/mean_shape.pth
COPY ./models/unet-rectangle-epoch=08-val_loss=0.14-unet-training-rectangle_1.ckpt /app/models/unet-rectangle-epoch=08-val_loss=0.14-unet-training-rectangle_1.ckpt

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONPATH="/app"
EXPOSE 7860
CMD ["python", "wings/app/app.py"]

FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install gradio lightning loguru opencv-python scipy scikit-learn matplotlib

WORKDIR /app
COPY ./wings /app/wings

EXPOSE 7860
CMD ["python", "wings/app/app.py"]

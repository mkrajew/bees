# Bee Wing Images

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Segmentation of bee wing images


## Running the Application

This project includes a `Makefile` that provides a simple command to start the app.

### **Run the App**

Use the following command:

```bash
make app
```


## Build & Run in Docker

### Build
```bash
docker build -t pytorch-gradio .
```

### Run (CPU mode)
```
docker run -p 7860:7860 pytorch-gradio
```

### Run with CUDA GPU (NVIDIA drivers required)
```
docker run --gpus all -p 7860:7860 pytorch-gradio
```
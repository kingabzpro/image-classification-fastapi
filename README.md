# Image Classification Inference with FastAPI: From Model to Production API

This repository demonstrates how to fine-tune an image classification model on the CIFAR10 dataset and deploy it as a production-ready API using FastAPI. The project covers end-to-end workflows—from training a model with PyTorch to serving real-time predictions via multiple API endpoints.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Training the Model](#training-the-model)
- [Running the Inference API](#running-the-inference-api)
- [API Endpoints](#api-endpoints)
- [Client Request Example](#client-request-example)
- [Docker Deployment (Optional)](#docker-deployment-optional)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project illustrates a complete pipeline for image classification:
- **Fine-tuning a Pre-Trained Model:** Adapt ResNet18 for the CIFAR10 dataset.
- **API Deployment with FastAPI:** Create a production-ready REST API for model inference.
- **Extended Functionality:** Endpoints for health check, model metadata, class names, and predictions (including confidence scores).

---

## Features

- **Model Fine-Tuning:** Train a ResNet18 model using CIFAR10 with proper data augmentation and normalization.
- **Production-Ready API:** Serve model predictions with FastAPI, utilizing multiple endpoints.
- **Additional Endpoints:** 
  - Health check (`/health`)
  - Model metadata (`/metadata`)
  - Class names (`/class_names`)
  - Standard prediction (`/predict`)
  - Prediction with confidence scores (`/predict_with_confidence`)
- **Client Example:** Sample Python client code using the `requests` library.
- **Docker Support:** (Optional) Containerize the API for scalable deployment.

---

## Project Structure

```plaintext
.
├── app.py                     # FastAPI application for inference
├── finetune_model.py          # Script to fine-tune the model on CIFAR10
├── client.py                  # Example client for API requests
├── requirements.txt           # Python dependencies
├── Dockerfile                 # (Optional) Docker configuration for deployment
└── README.md                  # This readme file
```

---

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Pillow](https://python-pillow.org/)
- [Requests](https://docs.python-requests.org/) (for client usage)

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/image-classification-fastapi.git
   cd image-classification-fastapi
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Docker Setup:**

   If you plan to deploy using Docker, ensure you have Docker installed.

---

## Training the Model

Fine-tune the ResNet18 model on the CIFAR10 dataset using the provided training script:

```bash
python finetune_model.py
```

This script:
- Downloads the CIFAR10 dataset.
- Applies necessary transformations.
- Fine-tunes a pre-trained ResNet18 model.
- Saves the trained weights to `finetuned_model.pth`.

---

## Running the Inference API

Start the FastAPI inference server with Uvicorn:

```bash
python app.py
```

The API will be available at `http://0.0.0.0:8000`. For production, the app is configured to use multiple workers.

---

## API Endpoints

### 1. Health Check

- **Endpoint:** `/health`
- **Method:** GET
- **Description:** Verify that the API is running and check the current device.
  
### 2. Model Metadata

- **Endpoint:** `/metadata`
- **Method:** GET
- **Description:** Retrieve model details (architecture, number of classes, class names, etc.).

### 3. Class Names

- **Endpoint:** `/class_names`
- **Method:** GET
- **Description:** Get the list of CIFAR10 class names.

### 4. Predict Image Class

- **Endpoint:** `/predict`
- **Method:** POST
- **Description:** Upload an image and receive the predicted class.
- **Request:** Multipart form data with an image file (PNG/JPEG).

### 5. Predict with Confidence

- **Endpoint:** `/predict_with_confidence`
- **Method:** POST
- **Description:** Returns the top-3 predicted classes with confidence scores.
- **Request:** Multipart form data with an image file (PNG/JPEG).

---

## Client Request Example

An example client (`client.py`) demonstrates how to interact with the API endpoints using Python's `requests` library. To run the client:

```bash
python client.py
```

The client performs:
- Health check
- Retrieval of metadata and class names
- Image predictions (with and without confidence scores)

---

## Docker Deployment (Optional)

To build and run the API in a Docker container:

1. **Build the Docker Image:**

   ```bash
   docker build -t image-classification-api .
   ```

2. **Run the Docker Container:**

   ```bash
   docker run -p 8000:8000 image-classification-api
   ```

The API will be available at `http://localhost:8000`.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any improvements, bug fixes, or additional features.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or inquiries, please contact [nexusml](https://nexusml.co/contact/).

---

Happy Coding!



import io

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="A production-ready API for image classification using a fine-tuned model on CIFAR10.",
)

# Define CIFAR10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned ResNet18 model.
# Replace the deprecated 'pretrained=False' with 'weights=None'
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load("finetuned_model.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing transforms (should match the validation transforms used during training)
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@app.get("/health", summary="Health Check", tags=["Status"])
async def health_check():
    return {"status": "ok", "message": "API is running", "device": str(device)}


@app.get("/metadata", summary="Get Model Metadata", tags=["Metadata"])
async def get_metadata():
    metadata = {
        "model_architecture": "ResNet18",
        "num_classes": num_classes,
        "class_names": class_names,
        "device": str(device),
        "model_weights_file": "finetuned_model.pth",
        "description": "Model fine-tuned on CIFAR10 dataset",
    }
    return JSONResponse(metadata)


@app.get("/class_names", summary="Get Class Names", tags=["Metadata"])
async def get_class_names():
    return JSONResponse({"class_names": class_names})


@app.post("/predict", summary="Predict Image Class", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    # Validate file type.
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only PNG and JPEG are supported.",
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Error processing image.")

    # Preprocess the image.
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    predicted_class = class_names[preds[0]]
    return JSONResponse({"predicted_class": predicted_class})


@app.post(
    "/predict_with_confidence",
    summary="Predict Image Class with Confidence",
    tags=["Inference"],
)
async def predict_with_confidence(file: UploadFile = File(...)):
    # Validate file type.
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only PNG and JPEG are supported.",
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Error processing image.")

    # Preprocess the image.
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, k=3)  # get top 3 predictions
        top_probs = top_probs.cpu().numpy().tolist()[0]
        top_idxs = top_idxs.cpu().numpy().tolist()[0]
        predictions = [
            {"class": class_names[idx], "confidence": prob}
            for idx, prob in zip(top_idxs, top_probs)
        ]

    return JSONResponse({"predictions": predictions})


if __name__ == "__main__":
    # Run the API with multiple workers for production readiness.
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=4)

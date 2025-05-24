import base64
import io
import json
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from torchvision.transforms import v2

# Define ImageNet stats (if not already defined globally)
# Make sure these values are correct for your training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Define image transforms - MUST MATCH VALIDATION TRANSFORMS
# Assuming the target size for the model is 224x224

# Add this after creating the FastAPI app instance (app = FastAPI(...))
app = FastAPI(title="State GeoGuesser API")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, you can use ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# List of US states (assuming 50 classes in your model)
STATE_LABELS = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming"
]

# Optional state abbreviations for concise output
STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size)
        self.mp = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


def build_head():
    head = nn.Sequential(
        # Fastai uses an AdaptiveConcatPool2d to combine max and avg pool outputs.
        AdaptiveConcatPool2d(output_size=1),  # output shape: [batch, fc_in*2, 1, 1]
        nn.Flatten(),  # flatten to shape: [batch, fc_in*2]
        nn.BatchNorm1d(4096, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(0.25),
        nn.Linear(4096, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(0.5),
        nn.Linear(512, 50, bias=False),
    )
    return head

def build_state_classifier(num_classes=50, pretrained=True, dropout_rate=0.0):
    """
    Build a ResNet101 model for state classification with AdaptiveConcatPool2d
    that preserves the original layer structure.
    """
    # Load pretrained model
    if pretrained:
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet101(weights=None)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of features
    fc_in = model.fc.in_features  # typically 2048 for resnet101

    # Define a custom forward function that ensures proper data flow
    def custom_forward(x, model):
        # Pass x through the body (up to layer4)
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        # Do NOT flatten x here; pass the 4D tensor directly to the head
        x = model.fc(x)
        return x

    # Replace the classifier head
    model.avgpool = nn.Identity()
    model.fc = build_head()
    # Override the forward method
    model.forward = lambda x: custom_forward(x, model)

    return model

# Create the model
model = build_state_classifier(pretrained=False)
# Load the model weights
try:
    weights = torch.load('./weights/best_model.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(weights['model_state'])
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Define image transforms - same as used during training
preprocess = v2.Compose([
    v2.ToImage(), # Convert PIL Image to torch Tensor
    v2.Resize(size=(224, 224), antialias=True), # Resize directly to 224x224
    v2.ToDtype(torch.float32, scale=True), # Convert to float32 and scale to [0.0, 1.0]
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # Normalize
])


@app.get("/")
async def root():
    return {"message": "State GeoGuesser API", "status": "online"}


@app.get("/model/status")
async def get_status():
    return {
        "message": "Model loaded",
        "model_type": "ResNet101",
        "num_classes": len(STATE_LABELS)
    }

async def process_single_image(img):
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.inference_mode():
        output = model(img_tensor)

    return output


def format_predictions(probabilities):
    """Format the prediction results."""
    # Get top 5 predictions
    topk_values, topk_indices = torch.topk(probabilities, 5)

    # Prepare results
    predictions = []
    for i, (prob, idx) in enumerate(zip(topk_values.tolist(), topk_indices.tolist())):
        predictions.append({
            "rank": i + 1,
            "state": STATE_LABELS[idx],
            "state_abbrev": STATE_ABBREV[STATE_LABELS[idx]],
            "probability": round(prob * 100, 2)
        })

    return {
        "predictions": predictions,
        "top_prediction": STATE_LABELS[topk_indices[0].item()],
        "top_prediction_abbrev": STATE_ABBREV[STATE_LABELS[topk_indices[0].item()]],
        "confidence": round(topk_values[0].item() * 100, 2)
    }

@app.post("/predict_panorama")
async def predict_panorama(
    north: UploadFile = File(...),
    east: UploadFile = File(...),
    south: UploadFile = File(...),
    west: UploadFile = File(...)):
    """Predicts the state from a four-view panorama."""

    for img_file in [north, east, south, west]:
        if not img_file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {img_file.filename} must be an image")

    try:
        outputs = []
        for img_file in [north, east, south, west]:
            content = await img_file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            output = await process_single_image(img)
            outputs.append(output)

        stacked_outputs = torch.cat(outputs, dim=0)
        avg_output = torch.mean(stacked_outputs, dim=0, keepdim=True)
        probabilities = torch.softmax(avg_output, dim=1)[0]

        return format_predictions(probabilities)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Panorama prediction error: {str(e)}")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Check if the file is an image
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process the image
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        # Preprocess the image
        output = await process_single_image(img)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]# Add batch dimension

        return format_predictions(probabilities)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_base64")
async def predict_base64(data: Dict):
    try:
        # Get base64 encoded image from request
        if "image" not in data:
            raise HTTPException(status_code=400, detail="Base64 encoded image required")

        base64_image = data["image"]

        # Decode and process the image
        img_bytes = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess the image
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        # Get top 5 predictions
        topk_values, topk_indices = torch.topk(probabilities, 5)

        # Prepare results
        predictions = []
        for i, (prob, idx) in enumerate(zip(topk_values.tolist(), topk_indices.tolist())):
            predictions.append({
                "rank": i + 1,
                "state": STATE_LABELS[idx],
                "probability": round(prob * 100, 2)
            })

        return {
            "predictions": predictions,
            "top_prediction": STATE_LABELS[topk_indices[0].item()],
            "confidence": round(topk_values[0].item() * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/test")
async def test_connection():
    return {
        "status": "success",
        "message": "Connection to GeoGuesser API successful"
    }
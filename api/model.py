import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Class names — order MUST match what ImageFolder produced in Module 4
class_names = ['cat', 'dog']

# Pick the device once at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(weights_path: str):
    """Recreate the ResNet18 architecture and load our saved weights into it."""
    # 1. Same architecture we trained — ResNet18 with a 2-class final layer
    model = models.resnet18(weights=None)              # no pretrained weights this time
    model.fc = nn.Linear(model.fc.in_features, 2)      # 2 = cat, dog

    # 2. Pour the saved weights into it
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    # 3. Move to GPU/CPU and switch to evaluation mode
    model.to(device)
    model.eval()
    return model

# Load it ONCE when this module is imported, not on every request
model = load_model('cats_vs_dogs_resnet18.pth')
print(f"Model loaded on {device}")

# Identical to test_transform in Module 4 — no augmentation, just resize + normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_bytes: bytes) -> dict:
    """Take raw image bytes, return prediction + confidence."""
    # 1. Open the image with PIL and force it to RGB
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # 2. Apply the same transforms as test time, then add a batch dimension
    tensor = transform(image).unsqueeze(0).to(device)

    # 3. Forward pass — no gradients needed
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    return {
        'prediction': class_names[pred_idx],
        'confidence': round(confidence, 4),
        'probabilities': {
            class_names[i]: round(probs[i].item(), 4)
            for i in range(len(class_names))
        }
    }
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

trained_model = None

class_names = [
    "Front Breakage",
    "Front Crushed",
    "Front Normal",
    "Rear Breakage",
    "Rear Crushed",
    "Rear Normal",
]


# ------------------ MODEL DEFINITION ------------------
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final FC layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ------------------ PREDICTION FUNCTION ------------------
def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = CarClassifierResNet()

        # ✅ Robust, cloud-safe model path
        MODEL_PATH = Path(__file__).parent / "model" / "saved_model.pth"

        # ✅ CPU-safe loading
        trained_model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        )

        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    return class_names[predicted_class.item()]

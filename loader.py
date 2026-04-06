from pathlib import Path
import torch

from model import BaseballCNN


def load_trained_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    classes = checkpoint["classes"]
    img_size = checkpoint["img_size"]

    model = BaseballCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes, img_size


if __name__ == "__main__":
    model_path = Path(__file__).resolve().parent / "saved_models" / "baseball_model.pt"

    model, classes, img_size = load_trained_model(model_path)

    print("Model loaded successfully")
    print("Classes:", classes)
    print("Image size:", img_size)

    x = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        pred = model(x)

    predicted_class = classes[pred.argmax(1).item()]
    print("Test prediction:", predicted_class)
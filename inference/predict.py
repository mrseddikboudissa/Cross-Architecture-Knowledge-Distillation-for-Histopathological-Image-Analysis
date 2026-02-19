import torch
from models.loadmodel import load_student_model
from data.transforms import load_image

from utils.config import load_config
config = load_config("configs/warmup.yaml")

MODEL_PATH = config["inference"]["model_path"]
DEVICE = config["inference"]["device"]

model = load_student_model(MODEL_PATH, config, DEVICE)  # ✅ pass config

def predict(image_path):
    image = load_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).item()

    return outputs

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    print(predict(image_path))
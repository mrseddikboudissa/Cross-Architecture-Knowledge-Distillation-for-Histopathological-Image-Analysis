import torch
from models.student import StudentModel

def load_student_model(model_path, cfg, device="cuda"):
    model = StudentModel(cfg)   # ✅ pass cfg here
    
    
    checkpoint = torch.load(model_path, map_location=device)

    print(list(checkpoint.keys())[:10])
    print(list(model.backbone.state_dict().keys())[:10])
    model.backbone.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model
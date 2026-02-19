import torch
from utils.layer_utils import split_layers_into_thirds


def extract_stage_activations(model, x, stage="stage1"):
    """
    Extracts activations from a specific third of the model
    """
    model.eval()
    outputs = []

    layers = split_layers_into_thirds(model)[stage]

    def hook_fn(module, input, output):
        outputs.append(output)

    hooks = [layer.register_forward_hook(hook_fn) for layer in layers]

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    return outputs
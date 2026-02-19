def get_leaf_layers(model):
    """
    Returns all leaf layers (no children)
    """
    return [layer for layer in model.modules() if len(list(layer.children())) == 0]


def split_layers_into_thirds(model):
    """
    Splits leaf layers into 3 equal parts
    """
    leaf_layers = get_leaf_layers(model)
    n = len(leaf_layers)
    k = n // 3

    return {
        "stage1": leaf_layers[:k],
        "stage2": leaf_layers[k:2 * k],
        "stage3": leaf_layers[2 * k:]
    }
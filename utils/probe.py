import torch


def build_probe_batch(
    dataloader,
    num_images,
    device
):
    """
    Collects a fixed number of images from a dataloader
    for representation comparison (CKA / KCCA).
    """
    images_list = []

    for images, _ in dataloader:
        images_list.append(images)
        total = sum(img.shape[0] for img in images_list)
        if total >= num_images:
            break

    images_big_batch = torch.cat(images_list, dim=0)[:num_images]
    return images_big_batch.to(device)
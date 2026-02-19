import numpy as np
from mvlearn.embed import KMCCA
from sklearn.decomposition import PCA


# ------------------------------------------------------------
# Feature standardization (CNN vs ViT)
# ------------------------------------------------------------

def process_cnn_features(feats):
    """
    feats: torch.Tensor (B, C, H, W)
    return: numpy array (B, C)
    """
    feats = feats.mean(dim=(2, 3))  # Global Average Pooling
    return feats.detach().cpu().numpy()


def process_vit_features(feats):
    """
    feats: torch.Tensor (B, N, D)
    return: numpy array (B, D)
    """
    feats = feats.mean(dim=1)  # Mean over tokens
    return feats.detach().cpu().numpy()


# ------------------------------------------------------------
# PCA (mandatory before KCCA)
# ------------------------------------------------------------

def apply_pca(X, n_components=32):
    """
    X: (B, D)
    """
    pca = PCA(n_components=n_components, whiten=True)
    return pca.fit_transform(X)


# ------------------------------------------------------------
# KCCA computation
# ------------------------------------------------------------

def compute_kcca(
    teacher_feats,
    student_feats,
    pca_dim=32,
    kcca_dim=10,
    gamma=None,
    reg=1e-3,
):
    """
    teacher_feats: torch.Tensor (CNN or ViT)
    student_feats: torch.Tensor (CNN or ViT)

    Returns:
        mean canonical correlation (float)
    """

    # --- Standardize representations ---
    if teacher_feats.dim() == 4:
        X = process_cnn_features(teacher_feats)
    else:
        X = process_vit_features(teacher_feats)

    if student_feats.dim() == 4:
        Y = process_cnn_features(student_feats)
    else:
        Y = process_vit_features(student_feats)

    # --- PCA ---
    X = apply_pca(X, pca_dim)
    Y = apply_pca(Y, pca_dim)

    # --- KCCA ---
    kcca = KMCCA(
        n_components=kcca_dim,
        kernel="rbf",
        kernel_params={"gamma": gamma},
        regs=reg,
    )

    U, V = kcca.fit_transform([X, Y])

    # --- Mean canonical correlation ---
    corrs = [
        np.corrcoef(U[:, i], V[:, i])[0, 1]
        for i in range(kcca_dim)
    ]

    return float(np.mean(corrs))
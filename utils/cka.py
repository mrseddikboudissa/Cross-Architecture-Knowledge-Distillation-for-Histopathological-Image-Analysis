import numpy as np
from utils.hooks import extract_stage_activations
from tqdm import tqdm

def unbiased_HSIC(K, L):
    n = K.shape[0]
    ones = np.ones(shape=(n))

    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)

    trace = np.trace(np.dot(K, L))

    nom1 = np.dot(np.dot(ones.T, K), ones)
    nom2 = np.dot(np.dot(ones.T, L), ones)
    denom = (n - 1) * (n - 2)
    middle = (nom1 * nom2) / denom

    last = (2 / (n - 2)) * np.dot(np.dot(ones.T, K), np.dot(L, ones))

    return (trace + middle - last) / (n * (n - 3))


def CKA(X, Y):
    nom = unbiased_HSIC(X @ X.T, Y @ Y.T)
    den1 = unbiased_HSIC(X @ X.T, X @ X.T)
    den2 = unbiased_HSIC(Y @ Y.T, Y @ Y.T)
    return nom / np.sqrt(den1 * den2)


def calculate_CKA_for_two_activations(actA, actB):
    if hasattr(actA, "detach"):
        actA = actA.detach().cpu().numpy()
    if hasattr(actB, "detach"):
        actB = actB.detach().cpu().numpy()

    actA = actA.reshape(actA.shape[0], -1)
    actB = actB.reshape(actB.shape[0], -1)

    return CKA(actA, actB)




def compute_stage_cka_matrix(modelA, modelB, x, stage="stage1"):
    actsA = extract_stage_activations(modelA, x, stage)
    actsB = extract_stage_activations(modelB, x, stage)

    cka_matrix = np.zeros((len(actsA), len(actsB)))

    for i, a in enumerate(tqdm(actsA, desc="Model A layers")):
        for j, b in enumerate(actsB):
            cka_matrix[i, j] = calculate_CKA_for_two_activations(a, b)

    return cka_matrix




def select_topk_pairs(cka_matrix, k):
    """
    Select top-k (student_layer, teacher_layer) pairs
    from a CKA similarity matrix.

    Args:
        cka_matrix (np.ndarray): shape (n_student_layers, n_teacher_layers)
        k (int): number of top pairs

    Returns:
        List of tuples: [(s_idx, t_idx, cka_score), ...]
    """
    n_s, n_t = cka_matrix.shape

    pairs = []
    for i in range(n_s):
        for j in range(n_t):
            pairs.append((i, j, cka_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    return pairs[:k]
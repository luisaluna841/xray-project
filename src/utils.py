import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Define semente global para reprodutibilidade.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(project_root):
    """
    Garante que as pastas models/ e outputs/ existam.
    """

    models_dir = os.path.join(project_root, "models")
    outputs_dir = os.path.join(project_root, "outputs")
    figures_dir = os.path.join(outputs_dir, "figures")
    gradcam_dir = os.path.join(figures_dir, "gradcam")
    metrics_dir = os.path.join(outputs_dir, "metrics")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    return models_dir, metrics_dir, gradcam_dir


import pickle

def save_metrics(metrics_dict, save_path):
    """
    Salva métricas de forma robusta e compatível
    usando pickle (.pkl).
    """

    if not save_path.endswith(".pkl"):
        save_path = save_path.replace(".npz", ".pkl")

    with open(save_path, "wb") as f:
        pickle.dump(metrics_dict, f)

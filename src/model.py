import torch.nn as nn
from torchvision import models


def get_model(model_name="resnet18", num_classes=2, pretrained=True):
    """
    Retorna o modelo escolhido e a camada alvo para Grad-CAM.

    Parâmetros:
        model_name (str): 
            - "resnet18"
            - "densenet121"
        num_classes (int): número de classes (binário → 2)
        pretrained (bool): usar pesos do ImageNet

    Retorno:
        model (nn.Module)
        target_layer (camada usada futuramente no Grad-CAM)
    """

    if model_name == "resnet18":

        model = models.resnet18(pretrained=pretrained)

        # Substituir camada final
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        # Camada alvo para Grad-CAM
        target_layer = model.layer4

    elif model_name == "densenet121":

        model = models.densenet121(pretrained=pretrained)

        # Substituir camada final
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

        # Camada alvo para Grad-CAM
        target_layer = model.features

    else:
        raise ValueError("Modelo não suportado. Use 'resnet18' ou 'densenet121'.")

    return model, target_layer

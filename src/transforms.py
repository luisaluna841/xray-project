from torchvision import transforms


def get_transforms(img_size=224, augmentation="light"):
    """
    Retorna as transformações de treino e validação.

    Parâmetros:
        img_size (int): tamanho final da imagem
        augmentation (str):
            - "light"  → baseline
            - "strong" → hipótese 2

    Retorno:
        train_transform, val_transform
    """

    # Transformações comuns de validação
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),

        # Remove bordas externas (reduz artefatos)
        transforms.CenterCrop(int(img_size * 0.9)),

        transforms.ToTensor(),

        # Normalização padrão ImageNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Baseline: augmentation leve
    if augmentation == "light":

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(int(img_size * 0.9)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Hipótese 2: augmentation mais forte
    elif augmentation == "strong":

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(int(img_size * 0.9)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:
        raise ValueError("augmentation deve ser 'light' ou 'strong'")

    return train_transform, val_transform

import torch
from torch.utils.data import Dataset
from PIL import Image


class XRayDataset(Dataset):
    """
    Dataset customizado para imagens de raio-X.

    Espera um DataFrame contendo:
        - path
        - label

    Parâmetros:
        dataframe (pd.DataFrame)
        transform (callable)
    """

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # Carregar imagem
        image = Image.open(row["path"]).convert("RGB")

        # Aplicar transformações
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label

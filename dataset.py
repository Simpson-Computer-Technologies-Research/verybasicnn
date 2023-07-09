from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd


# Functions for the datasets
class Datasets:
    @staticmethod
    def fromcsv(path: str) -> DataLoader:
        return DataLoader(
            CSVDataset(path=path), batch_size=64, shuffle=True)

    @staticmethod
    def test_one() -> torch.Tensor:
        return torch.tensor([[1]]).float()
    
    @staticmethod
    def test_zero() -> torch.Tensor:
        return torch.tensor([[0]]).float()


# Custom csv dataset
class CSVDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.csv_file: pd.DataFrame = pd.read_csv(path)

    def __len__(self) -> int:
        return len(self.csv_file)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.csv_file.iloc
        x = self.to_float(data[index, 0])
        y = self.to_float(data[index, 1])
        return torch.tensor([[x]]).float(), torch.tensor([[y]]).float()
    
    def to_float(self, value) -> float:
        return float(str(value))
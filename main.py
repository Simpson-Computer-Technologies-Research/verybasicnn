from dataset import Datasets
from torch.utils.data import DataLoader
from model import Model
import torch

# Create a model
model: Model = Model(1, 1)

# Create a dataset
dataset: DataLoader = Datasets.fromcsv("data.csv")

# Train the model
model.train(dataset, 1000)

# Save the model
model.save()

# Test the model
def test(data):
    pred: torch.Tensor = model(data)
    print(f"Prediction: {pred.item()}")

    # Determine whether the prediction is 1 or 0
    if pred.item() >= 0.5:
        print("1 == 1")
    else:
        print("0 == 0")
        

# Run the program
if __name__ == "__main__":
    test(Datasets.test_one())
    test(Datasets.test_zero())

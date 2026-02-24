
from dataset import SpotDataset

def main() -> int:

    dataset_path = "./dataset/spot"

    dataloader = SpotDataset.create_dataloader(root_dir=dataset_path)

import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from nvblox_torch.sensor import Sensor

class SpotDataset(Dataset):

    def __init__(self, root_dir: str, seq_name: str, device: str = "cuda") -> None:
        super().__init__()
        self.root_dir = root_dir
        self.device = device
        
        # Load all floats from camera intrinsics file
        # File has variable columns per row (3x3 matrix, then single values for distortion)
        intrinsics_path = os.path.join(self.root_dir, 'camera-intrinsics.txt')

        self.sequence_name = seq_name
        self.seq_dir = os.path.join(self.root_dir, self.sequence_name)

        self.frame_names = list(sorted({f.split('.')[0] for f in os.listdir(self.seq_dir)}))

        # Load first frame to determine image dimensions and create sensor
        # first_frame_name = self.frame_names[0]
        # first_rgb_np = self._load_color(first_frame_name)
        # height, width = first_rgb_np.shape[:2]

        # Create sensor object from camera intrinsics and distortion coefficients
        # self.sensor = Sensor.from_file(intrinsics_path, width, height)
        

    @staticmethod
    def create_dataloader(root_dir: str, seq_name: str) -> DataLoader:
        return DataLoader(SpotDataset(root_dir=root_dir, seq_name=seq_name),
                          batch_size=1,
                          shuffle=False,
                          num_workers=0,
                          collate_fn=collate_batch)
from typing import Tuple
from datasets import load_dataset
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

class CIFAR10HFDataset(Dataset):
    def __init__(self, split: str = 'train', image_size: int = 32):
        self.ds = load_dataset('uoft-cs/cifar10', split=split)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple:
        item = self.ds[idx]
        img_arr = item.get('img', None)
        if img_arr is None:
            img_arr = item.get('image')
        image = Image.fromarray(img_arr) if not isinstance(img_arr, Image.Image) else img_arr
        x = self.transform(image)
        y = item['label']
        return x, y

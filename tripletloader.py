import os
from typing import List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


class TripletDataset(Dataset):
    def __init__(self, images_dir: str, triplets_path: str, use_range: Tuple[int, int], transform=None,
                 file_extension: str = '.jpg'):
        self.images_dir = images_dir
        self.triplets_path = triplets_path

        self.transform = transform
        self.file_extension = file_extension

        self.triplet_image_paths: List[Tuple[str, str, str]] = []

        with open(triplets_path) as triplets_file:
            for line_i, line in enumerate(triplets_file):
                if line_i < use_range[0] or line_i >= use_range[1]:
                    continue

                line: str
                line = line.strip('\n')

                file_names = line.split(' ')
                assert len(file_names) == 3

                def get_path(n) -> str:
                    return os.path.join(images_dir, n + file_extension)

                self.triplet_image_paths.append((
                    get_path(file_names[0]),
                    get_path(file_names[1]),
                    get_path(file_names[2])
                ))

    def __getitem__(self, item: int):
        img_paths = self.triplet_image_paths[item]

        def load(p: str):
            img = Image.open(p)
            if self.transform is not None:
                img = self.transform(img)

            return img

        return (
            load(img_paths[0]),
            load(img_paths[1]),
            load(img_paths[2]),
            0  # Similarity condition index, not used here
        )

    def __len__(self):
        return len(self.triplet_image_paths)

import os
import random
import time
from argparse import ArgumentParser
from typing import Optional, List

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, ToTensor, Resize, CenterCrop

import embedmodule
from Resnet_18 import ResNet
from csn import ConditionalSimNet
from tripletnet import CS_Tripletnet


class ImgDataset(Dataset):
    def __init__(self, names: List[str], paths: List[str], transform):
        assert len(names) == len(paths)

        self.names = names
        self.paths = paths
        self.transform = transform

    def __getitem__(self, item):
        n = self.names[item]
        p = self.paths[item]

        img = Image.open(p)
        return n, p, self.transform(img)

    def __len__(self):
        return len(self.names)


def save_masks(parameters_path: str, images_dir: str, save_dir: str,
               shuffle: bool, count: Optional[int], batch_size: int,
               num_masks: int, embedding_size: int, learned_masks: bool, disjoint_masks: bool,
               use_gpu: bool):
    assert os.path.exists(images_dir)
    assert count is None or count > 0

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    image_names = os.listdir(images_dir)
    if shuffle:
        random.shuffle(image_names)
    if count is not None:
        image_names = image_names[0:count]
    image_paths = [os.path.join(images_dir, n) for n in image_names]

    model: CS_Tripletnet = embedmodule.TripletEmbedModule.create_model(
        num_masks=num_masks,
        embedding_size=embedding_size,
        learned_masks=learned_masks,
        disjoint_masks=disjoint_masks,
        use_gpu=use_gpu
    )
    model.load_state_dict(torch.load(parameters_path))
    model.eval()

    model.embeddingnet: ConditionalSimNet
    model.embeddingnet.embeddingnet: ResNet

    transform = Compose([
        Resize(112),
        CenterCrop(112),
        ToTensor(),
        Normalize(mean=embedmodule.NORM_MEAN, std=embedmodule.NORM_STD)
    ])

    loader = DataLoader(
        ImgDataset(image_names, image_paths, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )

    full_embeddings_path = os.path.join(save_dir, 'full_embeddings.csv')
    masked_embeddings_paths = [os.path.join(save_dir, 'mask_{}_embeddings.csv'.format(i + 1)) for i in range(num_masks)]

    def save_embeddings(save_path: str, names: List[str], embeddings: torch.Tensor):
        with open(save_path, 'a') as f:
            for i, n in enumerate(names):
                line = ', '.join([n] + [str(v.item()) for v in embeddings[i]])
                f.write(line + '\n')

    start_time = time.time()
    cur_count = 0
    with torch.no_grad():
        for batch_i, data in enumerate(loader):
            n, p, img_t = data
            if use_gpu:
                img_t = img_t.cuda()

            full_em = model.embeddingnet.embeddingnet(img_t).to('cpu')
            save_embeddings(full_embeddings_path, n, full_em)

            for mask_i in range(num_masks):
                mask_t = torch.full((img_t.shape[0],), fill_value=mask_i, dtype=torch.long)
                if use_gpu:
                    mask_t = mask_t.cuda()
                m = model.embeddingnet(img_t, mask_t)[0].to('cpu')
                save_embeddings(masked_embeddings_paths[mask_i], n, m)

            cur_count += img_t.shape[0]
            print('\r{} / {} ({:.2f}s)'.format(cur_count, count, time.time() - start_time), end='')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--parameters', '-p', type=str, required=True, help='Path to model parameters.')
    parser.add_argument('--images-dir', '-i', type=str, required=True, help='Directory containing images.')
    parser.add_argument('--save-dir', '-s', type=str, required=True, help='Directory in which to save masks.')
    parser.add_argument('--no-shuffle', '-ns', action='store_true', help='Disable random shuffle of images.')
    parser.add_argument('--count', '-c', type=int, default=None, help='Number of images to save masks for.')
    parser.add_argument('--batch-size', '-b', type=int, default=100)

    parser.add_argument('--num-masks', '--nmasks', type=int, default=4)
    parser.add_argument('--embedding-size', '--esize', type=int, default=64)
    parser.add_argument('--learned_masks', '-lm', action='store_true')
    parser.add_argument('--disjoint_masks', '-dm', action='store_true')

    parser.add_argument('--use-gpu', '--gpu', action='store_true')

    args = parser.parse_args()
    print(args)

    args.use_gpu = args.use_gpu and torch.cuda.is_available()

    save_masks(
        parameters_path=args.parameters,
        images_dir=args.images_dir,
        save_dir=args.save_dir,
        shuffle=not args.no_shuffle,
        count=args.count,
        batch_size=args.batch_size,
        num_masks=args.num_masks,
        embedding_size=args.embedding_size,
        learned_masks=args.learned_masks,
        disjoint_masks=args.disjoint_masks,
        use_gpu=args.use_gpu
    )

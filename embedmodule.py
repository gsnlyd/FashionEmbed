import math
import os
from typing import Tuple, List

import torch
from pytorch_lightning.logging import TestTubeLogger
from torch import Tensor
from pytorch_lightning import LightningModule, data_loader
from test_tube import HyperOptArgumentParser
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip

import Resnet_18
from csn import ConditionalSimNet
from triplet_image_loader import TripletImageLoader
from tripletloader import TripletDataset
from tripletnet import CS_Tripletnet

DEFAULT_IMAGES_DIR = 'images'
DEFAULT_TRIPLETS_FILE_NAME = 'triplets.txt'

PARAMETERS_DIR_NAME = 'parameters'

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
NORM_STD = torch.tensor([0.229, 0.224, 0.225])


class TripletEmbedModule(LightningModule):
    def __init__(self, hparams):
        super(TripletEmbedModule, self).__init__()

        self.hparams = hparams
        self.batch_size = hparams.batch_size

        self.tripletnet = self.create_model(
            num_masks=hparams.num_masks,
            embedding_size=hparams.embedding_size,
            learned_masks=hparams.learned_masks,
            disjoint_masks=hparams.disjoint_masks,
            use_gpu=hparams.use_gpu
        )

        self.criterion = torch.nn.MarginRankingLoss(margin=hparams.margin)

        self.normalize = Normalize(NORM_MEAN.tolist(), NORM_STD.tolist())
        self.denormalize = Normalize((-NORM_MEAN / NORM_STD).tolist(), (1.0 / NORM_STD).tolist())

    @staticmethod
    def create_model(num_masks: int, embedding_size: int, learned_masks: bool,
                     disjoint_masks: bool, use_gpu: bool) -> CS_Tripletnet:
        embed_model = Resnet_18.resnet18(pretrained=True, embedding_size=embedding_size)
        csn_model = ConditionalSimNet(embed_model,
                                      n_conditions=num_masks,
                                      embedding_size=embedding_size,
                                      learnedmask=learned_masks,
                                      prein=disjoint_masks)
        tripletnet: CS_Tripletnet = CS_Tripletnet(csn_model,
                                                  num_concepts=num_masks,
                                                  use_cuda=use_gpu)
        if use_gpu:
            tripletnet.cuda()

        return tripletnet

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.tripletnet.parameters())
        optimizer = Adam(parameters, self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=(1 - 0.015))

        return [optimizer], [scheduler]

    DistsAndNorm = Tuple[Tensor, Tensor, Tensor, Tensor]
    WeightedEmbeds = Tuple[Tensor, Tensor, Tensor]
    MaskedEmbeds = Tuple[List[Tensor], List[Tensor], List[Tensor]]

    def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Tuple[DistsAndNorm, WeightedEmbeds, MaskedEmbeds]:
        # x = anchor image, y = far image, z = close image
        dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm, embeddings, masked_embeddings = self.tripletnet(
            x, y, z, 0)

        return (dist_a, dist_b, mask_norm, embed_norm), embeddings, masked_embeddings

    def loss(self, dist_a: Tensor, dist_b: Tensor, mask_norm: Tensor, embed_norm: Tensor):
        target = torch.full(dist_a.shape, fill_value=1, requires_grad=True)
        if self.hparams.use_gpu:
            target = target.cuda()

        loss_triplet = self.criterion(dist_a, dist_b, target)
        loss_mask = mask_norm / self.batch_size
        loss_embed = embed_norm / math.sqrt(self.batch_size)

        return loss_triplet + (self.hparams.embed_loss * loss_embed) + (self.hparams.mask_loss * loss_mask)

    def training_step(self, batch, batch_idx):
        x, y, z, _ = batch

        output, embeddings, masked_embeddings = self.forward(x, y, z)
        loss_value = self.loss(*output)

        log_dict = {
            'train_loss': loss_value
        }

        return {
            'loss': loss_value,
            'progress_bar': log_dict,
            'log': log_dict
        }

    def validation_step(self, batch, batch_idx):
        x, y, z, _ = batch

        output, embeddings, masked_embeddings = self.forward(x, y, z)
        loss_value = self.loss(*output)

        dist_a, dist_b = output[0], output[1]
        # Accuracy is the fraction of triplets where the "far" image is closer
        # to the anchor than the "close" image
        accuracy = torch.sum(dist_a > dist_b).item() / self.batch_size

        return {
            'val_loss': loss_value,
            'val_accuracy': accuracy
        }

    def validation_end(self, outputs):
        def find_avg(key: str) -> float:
            return sum(o[key] for o in outputs) / len(outputs)

        avg_val_loss = find_avg('val_loss')
        avg_val_accuracy = find_avg('val_accuracy')

        log_dict = {
            'avg_val_loss': avg_val_loss,
            'avg_val_accuracy': avg_val_accuracy
        }

        return {
            'val_loss': avg_val_loss,
            'progress_bar': log_dict,
            'log': log_dict
        }

    def __log_embeddings(self):
        img_tensors = torch.tensor([])
        embeddings = torch.tensor([])
        masked_embeddings = [torch.tensor([]) for i in range(self.hparams.num_masks)]
        labels = []

        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader()[0]):
                if batch_idx * self.batch_size > self.hparams.num_embed_triplets:
                    break
                x, y, z, _ = batch

                if self.hparams.use_gpu:
                    x_cuda, y_cuda, z_cuda = x.cuda(), y.cuda(), z.cuda()
                else:
                    x_cuda, y_cuda, z_cuda = x, y, z

                _, output_embeddings, output_masked_embeddings = self.forward(x_cuda, y_cuda, z_cuda)

                output_embeddings = tuple(e.to('cpu') for e in output_embeddings)
                output_masked_embeddings = tuple([e.to('cpu') for e in l] for l in output_masked_embeddings)

                imgs = torch.cat([x, y, z], dim=0)
                imgs = torch.stack([self.denormalize(t) for t in imgs])
                output_embeddings = torch.cat(output_embeddings, dim=0)

                img_tensors = torch.cat([img_tensors, imgs], dim=0)
                embeddings = torch.cat([embeddings, output_embeddings], dim=0)

                for mask_i, (x_e, y_e, z_e) in enumerate(zip(*output_masked_embeddings)):
                    m_embeddings = torch.cat([x_e, y_e, z_e], dim=0)
                    masked_embeddings[mask_i] = torch.cat([masked_embeddings[mask_i], m_embeddings], dim=0)

                labels += (['x'] * self.batch_size + ['y'] * self.batch_size + ['z'] * self.batch_size)

        self.logger: TestTubeLogger
        for mask_i, em in enumerate([embeddings] + masked_embeddings):
            if mask_i == 0:
                tag = 'Weighted Embeddings'
            else:
                tag = 'Mask {} Embeddings'.format(mask_i)

            self.logger.experiment.add_embedding(
                mat=em,
                metadata=labels,
                label_img=img_tensors,
                global_step=self.global_step,
                tag=tag
            )

    def __save_parameters(self):
        self.logger: TestTubeLogger
        exp = self.logger.experiment

        exp_dir = exp.get_data_path(exp.name, exp.version)
        parameters_dir = os.path.join(exp_dir, PARAMETERS_DIR_NAME)

        if not os.path.exists(parameters_dir):
            os.mkdir(parameters_dir)

        save_name = 'version_{}_epoch_{}.pt'.format(exp.version, self.current_epoch)

        self.tripletnet: CS_Tripletnet
        torch.save(self.tripletnet.state_dict(),
                   os.path.join(parameters_dir, save_name))

    def on_epoch_end(self):
        self.__log_embeddings()
        self.__save_parameters()

    def __get_transforms(self, augment: bool):
        transforms = [
            Resize(112),
            CenterCrop(112),
            ToTensor(),
            self.normalize
        ]

        if augment:
            transforms.insert(2, RandomHorizontalFlip())

        return transforms

    def __make_dataloader_from_dataset(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

    def __make_default_dataloader(self, split: str, augment: bool, num_triplets: int):
        transforms = self.__get_transforms(augment)

        dataset = TripletImageLoader(
            root='data',
            base_path='ut-zap50k-images',
            filenames_filename='filenames.json',
            conditions=[0, 1, 2, 3],
            split=split,
            n_triplets=num_triplets,
            transform=Compose(transforms)
        )

        return self.__make_dataloader_from_dataset(dataset)

    def __make_configurable_dataloader(self, augment: bool, use_range: Tuple[int, int]):
        transforms = self.__get_transforms(augment)

        dataset = TripletDataset(
            images_dir=os.path.join(self.hparams.dataset, DEFAULT_IMAGES_DIR),
            triplets_path=os.path.join(self.hparams.dataset, DEFAULT_TRIPLETS_FILE_NAME),
            use_range=use_range,
            transform=Compose(transforms)
        )

        return self.__make_dataloader_from_dataset(dataset)

    @data_loader
    def train_dataloader(self):
        if self.hparams.dataset is not None:
            return self.__make_configurable_dataloader(augment=True, use_range=(0, self.hparams.num_train_triplets))

        return self.__make_default_dataloader(
            split='train',
            augment=True,
            num_triplets=self.hparams.num_train_triplets
        )

    @data_loader
    def val_dataloader(self):
        if self.hparams.dataset is not None:
            start = self.hparams.num_train_triplets
            end = start + self.hparams.num_val_triplets

            return self.__make_configurable_dataloader(augment=True, use_range=(start, end))

        return self.__make_default_dataloader(
            split='val',
            augment=False,
            num_triplets=self.hparams.num_val_triplets
        )

    @staticmethod
    def configure_parser(parser: HyperOptArgumentParser):
        parser.add_argument('--batch-size', '-b', type=int, default=96)
        parser.add_argument('--epochs', '-e', type=int, default=15)
        parser.add_argument('--learning-rate', '-lr', type=float, default=5e-5)
        parser.add_argument('--use-gpu', '--gpu', action='store_true')
        parser.add_argument('--resume', '-r', type=int, default=-1,
                            help='Resume training from a previous version.')

        parser.add_argument('--dataset', '-d', type=str, help='Path to optional custom dataset.')

        parser.add_argument('--num-masks', '--nmasks', type=int, default=4)
        parser.add_argument('--embedding-size', '--esize', type=int, default=64)
        parser.add_argument('--learned_masks', '-lm', action='store_true')
        parser.add_argument('--disjoint_masks', '-dm', action='store_true')

        parser.add_argument('--margin', '-m', type=float, default=0.2,
                            help='Triplet loss margin')
        parser.add_argument('--embed-loss', type=float, default=5e-3,
                            help='Loss multiplier for the embed norm')
        parser.add_argument('--mask-loss', type=float, default=5e-4,
                            help='Loss multiplier for the mask norm')

        parser.add_argument('--num_train_triplets', '--ntrain', type=int, default=100000)
        parser.add_argument('--num_val_triplets', '--nval', type=int, default=50000)
        parser.add_argument('--num_test_triplets', '--ntest', type=int, default=100000)
        parser.add_argument('--num_embed_triplets', '--nembed', type=int, default=100,
                            help='Number of triplets to add to the Embedding Projector after each epoch')

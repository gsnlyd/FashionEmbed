import math
from typing import Tuple

import torch
from pytorch_lightning.logging import TestTubeLogger
from torch import Tensor
from pytorch_lightning import LightningModule, data_loader
from test_tube import HyperOptArgumentParser
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip

import Resnet_18
from csn import ConditionalSimNet
from triplet_image_loader import TripletImageLoader
from tripletnet import CS_Tripletnet


class TripletEmbedModule(LightningModule):
    def __init__(self, hparams):
        super(TripletEmbedModule, self).__init__()

        self.hparams = hparams
        self.batch_size = hparams.batch_size

        embed_model = Resnet_18.resnet18(pretrained=True, embedding_size=hparams.embedding_size)
        csn_model = ConditionalSimNet(embed_model,
                                      n_conditions=hparams.num_masks,
                                      embedding_size=hparams.embedding_size,
                                      learnedmask=hparams.learned_masks,
                                      prein=hparams.disjoint_masks)
        self.tripletnet: CS_Tripletnet = CS_Tripletnet(csn_model,
                                                       num_concepts=hparams.num_masks,
                                                       use_cuda=hparams.use_gpu)
        if hparams.use_gpu:
            self.tripletnet.cuda()

        self.criterion = torch.nn.MarginRankingLoss(margin=hparams.margin)

        self.training_started = False

        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        self.normalize = Normalize(mean.tolist(), std.tolist())
        self.denormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.tripletnet.parameters())
        optimizer = Adam(parameters, self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=(1 - 0.015))

        return [optimizer], [scheduler]

    def forward(self, x: Tensor, y: Tensor, z: Tensor) -> \
            Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # x = anchor image, y = far image, z = close image
        dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm, embeddings = self.tripletnet(x, y, z, 0)

        return (dist_a, dist_b, mask_norm, embed_norm), embeddings

    def loss(self, dist_a: Tensor, dist_b: Tensor, mask_norm: Tensor, embed_norm: Tensor):
        target = torch.full(dist_a.shape, fill_value=1, requires_grad=True)
        if self.hparams.use_gpu:
            target = target.cuda()

        loss_triplet = self.criterion(dist_a, dist_b, target)
        loss_mask = mask_norm / self.batch_size
        loss_embed = embed_norm / math.sqrt(self.batch_size)

        return loss_triplet + (self.hparams.embed_loss * loss_embed) + (self.hparams.mask_loss * loss_mask)

    def training_step(self, batch, batch_idx):
        self.training_started = True

        x, y, z, c = batch

        output, embeddings = self.forward(x, y, z)
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
        x, y, z, c = batch

        output, embeddings = self.forward(x, y, z)
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

    def on_epoch_end(self):
        img_tensors = torch.tensor([])
        embeddings = torch.tensor([])
        labels = []

        for batch_idx, batch in enumerate(self.val_dataloader()[0]):
            if batch_idx * self.batch_size > self.hparams.num_embed_triplets:
                break
            x, y, z, c = batch

            if self.hparams.use_gpu:
                x, y, z = x.cuda(), y.cuda(), z.cuda()

            _, output_embeddings = self.forward(x, y, z)
            imgs = torch.cat([x, y, z], dim=0)
            imgs = torch.stack([self.denormalize(t) for t in imgs])
            output_embeddings = torch.cat(output_embeddings, dim=0)

            img_tensors = torch.cat([img_tensors, imgs], dim=0)
            embeddings = torch.cat([embeddings, output_embeddings], dim=0)

            labels += (['x'] * self.batch_size + ['y'] * self.batch_size + ['z'] * self.batch_size)

        self.logger: TestTubeLogger
        self.logger.experiment.add_embedding(
            mat=embeddings,
            metadata=labels,
            label_img=img_tensors,
            global_step=self.global_step,
            tag='Embeddings'
        )

    def __make_dataloader(self, split: str, augment: bool, num_triplets: int):
        transforms = [
            Resize(112),
            CenterCrop(112),
            ToTensor(),
            self.normalize
        ]

        if augment:
            transforms.insert(2, RandomHorizontalFlip())

        dataset = TripletImageLoader(
            root='data',
            base_path='ut-zap50k-images',
            filenames_filename='filenames.json',
            conditions=[0, 1, 2, 3],
            split=split,
            n_triplets=num_triplets,
            transform=Compose(transforms)
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

    @data_loader
    def train_dataloader(self):
        return self.__make_dataloader(
            split='train',
            augment=True,
            num_triplets=self.hparams.num_train_triplets
        )

    @data_loader
    def test_dataloader(self):
        return self.__make_dataloader(
            split='test',
            augment=False,
            num_triplets=self.hparams.num_test_triplets
        )

    @data_loader
    def val_dataloader(self):
        return self.__make_dataloader(
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

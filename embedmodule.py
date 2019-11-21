import torch
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
    def __init__(self, hparams: HyperOptArgumentParser):
        super(TripletEmbedModule, self).__init__()

        self.hparams = hparams
        self.batch_size = hparams.batch_size

        embed_model = Resnet_18.resnet18(pretrained=True, embedding_size=hparams.embedding_size)
        csn_model = ConditionalSimNet(embed_model,
                                      n_conditions=hparams.num_masks,
                                      embedding_size=hparams.embedding_size,
                                      learnedmask=hparams.learned_mask,
                                      prein=hparams.disjoint_mask)
        self.tripletnet: CS_Tripletnet = CS_Tripletnet(csn_model,
                                                       num_concepts=hparams.num_masks,
                                                       use_cuda=self.on_gpu)
        if self.on_gpu:
            self.tripletnet.cuda()

        self.criterion = torch.nn.MarginRankingLoss(margin=hparams.margin)

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.tripletnet.parameters())
        optimizer = Adam(parameters, self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=(1 - 0.015))

        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        pass

    def loss(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_end(self, outputs):
        pass

    def __make_dataloader(self, split: str, augment: bool, num_triplets: int):
        transforms = [
            Resize(112),
            CenterCrop(112),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        parser.add_argument('--num-masks', '--nmasks', type=int, default=4)
        parser.add_argument('--embedding-size', '--esize', type=int, default=64)

        parser.add_argument('--margin', '-m', type=int, default=15,
                            help='Triplet loss margin')
        parser.add_argument('--embed-loss', type=float, default=5e-3,
                            help='Loss multiplier for the embed norm')
        parser.add_argument('--mask-loss', type=float, default=5e-4,
                            help='Loss multiplier for the mask norm')

        parser.add_argument('--num_train_triplets', '--ntrain', default=100000)
        parser.add_argument('--num_val_triplets', '--nval', default=50000)
        parser.add_argument('--num_test_triplets', '--ntest', default=100000)

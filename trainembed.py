import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from test_tube import HyperOptArgumentParser

from embedmodule import TripletEmbedModule

DEFAULT_NAME = 'lightning_logs'


if __name__ == '__main__':
    parser = HyperOptArgumentParser()
    TripletEmbedModule.configure_parser(parser)
    hparams = parser.parse_args()
    hparams.use_gpu = hparams.use_gpu and torch.cuda.is_available()
    print(hparams)

    gpus = 1 if hparams.use_gpu else None

    if hparams.resume > -1:
        logger = TestTubeLogger(
            save_dir=os.getcwd(),
            name='lightning_logs',
            version=hparams.resume
        )
    else:
        logger = True

    model = TripletEmbedModule(hparams)
    trainer = Trainer(logger=logger,
                      min_nb_epochs=hparams.epochs,
                      gpus=gpus)

    trainer.fit(model)

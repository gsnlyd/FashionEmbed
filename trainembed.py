import torch
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser

from embedmodule import TripletEmbedModule

if __name__ == '__main__':
    parser = HyperOptArgumentParser()
    TripletEmbedModule.configure_parser(parser)
    hparams = parser.parse_args()
    hparams.use_gpu = hparams.use_gpu and torch.cuda.is_available()
    print(hparams)

    gpus = 1 if hparams.use_gpu else None

    model = TripletEmbedModule(hparams)
    trainer = Trainer(min_nb_epochs=hparams.epochs, gpus=gpus)

    trainer.fit(model)

from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser

from embedmodule import TripletEmbedModule

if __name__ == '__main__':
    parser = HyperOptArgumentParser()
    TripletEmbedModule.configure_parser(parser)
    hparams = parser.parse_args()
    print(hparams)

    model = TripletEmbedModule(hparams)
    trainer = Trainer(min_nb_epochs=hparams.epochs)

    trainer.fit(model)

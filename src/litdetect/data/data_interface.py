import importlib
import inspect
import traceback

import hydra
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from litdetect.scripts_init import get_logger

logger = get_logger(__file__)


class DataInterface(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.trainset = None
        self.valset = None
        self.testset = None
        self.predictset = None
        self.collate_fn = None

    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        logger.info(f'stage is {stage}, set dataset')
        if stage == 'fit':
            self.trainset, self.collate_fn = self.instancialize(data_mode='train', dataset=self.hparams.dataset)
            self.valset, _ = self.instancialize(data_mode='val', dataset=self.hparams.dataset)
        elif stage == 'validate':
            self.valset, self.collate_fn = self.instancialize(data_mode='val', dataset=self.hparams.dataset)
        elif stage == 'test':
            self.testset, self.collate_fn = self.instancialize(data_mode='test', dataset=self.hparams.dataset)
        elif stage == 'predict':
            self.predictset, self.collate_fn = self.instancialize(data_mode='pred', dataset=self.hparams.dataset)

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers, shuffle=True,
            pin_memory=True, collate_fn=self.collate_fn, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, shuffle=False,
            pin_memory=True, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, shuffle=False,
            pin_memory=True, collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predictset, batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers, shuffle=False,
            pin_memory=True, collate_fn=self.collate_fn
        )

    def instancialize(self, dataset, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        dataset_module = '.'.join(dataset.split('.')[:-1])
        dataset_name = dataset.split('.')[-1]

        try:
            data_module = getattr(importlib.import_module(
                dataset_module, package=__package__), dataset_name)
        except Exception as e:
            logger.error(f'Import Error: Failed to import \"{dataset_name}\" from \"{dataset_module}\"')
            traceback.print_exc()
            raise e

        class_args = inspect.getfullargspec(data_module.__init__).args[1:]

        args = {arg: self.hparams[arg] for arg in class_args if arg in self.hparams.keys()}
        args.update(other_args)

        transforms = hydra.utils.instantiate(
            self.hparams.augmentation_train
            if other_args['data_mode'] == 'train'
            else self.hparams.augmentation_val)

        try:
            collate_fn = getattr(importlib.import_module(
                dataset_module, package=__package__), 'collate_fn')
            logger.info('collate_fn was successfully loaded.')
        except ImportError:
            collate_fn = None
            logger.info('collate_fn not found! Use default collate_fn.')

        return data_module(transforms=transforms, **args), collate_fn

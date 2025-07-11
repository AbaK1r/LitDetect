import importlib
import inspect
import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataInterface(pl.LightningDataModule):
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
        logging.info(f'stage is {stage}, set dataset')
        if stage == 'fit':
            self.trainset, self.collate_fn = self.instancialize(data_mode='train', dataset=self.hparams.dataset)
            self.valset, _ = self.instancialize(data_mode='val', dataset=self.hparams.dataset)
        elif stage == 'validate':
            self.valset, self.collate_fn = self.instancialize(data_mode='val', dataset=self.hparams.dataset)
        elif stage == 'test':
            self.testset, self.collate_fn = self.instancialize(data_mode='val', dataset=self.hparams.dataset)
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
        """  # real_colon RealColon
        camel_name = ''.join([i.capitalize() for i in dataset.split('_')])
        # print(camel_name)
        try:
            data_module = getattr(importlib.import_module(
                '.' + dataset, package=__package__), camel_name)
            # data_module = getattr(importlib.import_module('.real_colon', package='data'), 'RealColon')
            # data_module = RealColon  # data_module赋值为一个class名
            # a = data_module()  # 类的实例化
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{dataset}.{camel_name}')

        class_args = inspect.getfullargspec(data_module.__init__).args[1:]

        args = {arg: self.hparams[arg] for arg in class_args if arg in self.hparams.keys()}
        args.update(other_args)

        try:
            collate_fn = getattr(importlib.import_module(
                '.' + dataset, package=__package__), 'collate_fn')
            logging.info('collate_fn was successfully loaded.')
        except:
            collate_fn = None
            logging.info('collate_fn not found! Use default collate_fn.')

        return data_module(**args), collate_fn

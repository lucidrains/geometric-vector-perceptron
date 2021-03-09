from argparse import ArgumentParser
from typing import List, Optional
from typing import Union

import numpy as np
import pytorch_lightning as pl
import sidechainnet
from sidechainnet.dataloaders.collate import get_collate_fn
from sidechainnet.utils.sequence import ProteinVocabulary
from torch.utils.data import DataLoader, Dataset


class ScnDataset(Dataset):
    def __init__(self, dataset, max_len: int):
        super(ScnDataset, self).__init__()
        self.dataset = dataset

        self.max_len = max_len
        self.scn_collate_fn = get_collate_fn(False)
        self.vocab = ProteinVocabulary()

    def collate_fn(self, batch):
        batch = self.scn_collate_fn(batch)
        real_seqs = [
            "".join([self.vocab.int2char(aa) for aa in seq])
            for seq in batch.int_seqs.numpy()
        ]
        seq = real_seqs[0][: self.max_len]
        true_coords = batch.crds[0].view(-1, 14, 3)[: self.max_len].view(-1, 3)
        angles = batch.angs[0, : self.max_len]
        mask = batch.msks[0, : self.max_len]

        # get padding
        padding_seq = (np.array([*seq]) == "_").sum()
        return {
            "seq": seq,
            "true_coords": true_coords,
            "angles": angles,
            "padding_seq": padding_seq,
            "mask": mask,
        }

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


class ScnDataModule(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--casp_version", type=int, default=7)
        parser.add_argument("--scn_dir", type=str, default="./sidechainnet_data")
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--eval_batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--train_max_len", type=int, default=256)
        parser.add_argument("--eval_max_len", type=int, default=256)

        return parser

    def __init__(
        self,
        casp_version: int = 7,
        scn_dir: str = "./sidechainnet_data",
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        num_workers: int = 1,
        train_max_len: int = 256,
        eval_max_len: int = 256,
        **kwargs,
    ):
        super().__init__()

        assert train_batch_size == eval_batch_size == 1, "batch size must be 1 for now"

        self.casp_version = casp_version
        self.scn_dir = scn_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_max_len = train_max_len
        self.eval_max_len = eval_max_len

    def setup(self, stage: Optional[str] = None):
        dataloaders = sidechainnet.load(
            casp_version=self.casp_version,
            scn_dir=self.scn_dir,
            with_pytorch="dataloaders",
        )
        print(
            dataloaders.keys()
        )  # ['train', 'train_eval', 'valid-10', ..., 'valid-90', 'test']

        self.train = ScnDataset(dataloaders["train"].dataset, self.train_max_len)
        self.val = ScnDataset(dataloaders["valid-90"].dataset, self.eval_max_len)
        self.test = ScnDataset(dataloaders["test"].dataset, self.eval_max_len)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.train.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.val.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.test.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    dm = ScnDataModule()
    dm.setup()

    train = dm.train_dataloader()
    print("train length", len(train))

    valid = dm.val_dataloader()
    print("valid length", len(valid))

    test = dm.test_dataloader()
    print("test length", len(test))

    for batch in test:
        print(batch)
        break

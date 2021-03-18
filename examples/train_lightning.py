import gc
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from loguru import logger
from pytorch_lightning.callbacks import (
    GPUStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from examples.data_handler import kabsch_torch, scn_cloud_mask
from examples.data_utils import (
    encode_whole_bonds,
    encode_whole_protein,
    from_encode_to_pred,
    prot_covalent_bond,
)
from examples.scn_data_module import ScnDataModule
from geometric_vector_perceptron.geometric_vector_perceptron import GVP_Network


class StructureModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        # model
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--cutoffs", type=float, default=1.0)
        parser.add_argument("--noise", type=float, default=1.0)

        # optimizer & scheduler
        parser.add_argument("--init_lr", type=float, default=1e-3)

        return parser

    def __init__(
        self,
        depth: int = 1,
        cutoffs: float = 1.0,
        noise: float = 1.0,
        init_lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        
        self.needed_info = {
            "cutoffs": [cutoffs], # -1e-3 for just covalent, "30_closest", 5. for under 5A, etc
            "bond_scales": [1, 2, 4],
            "aa_pos_scales": [1, 2, 4, 8, 16, 32, 64, 128],
            "atom_pos_scales": [1, 2, 4, 8, 16, 32],
            "dist2ca_norm_scales": [1, 2, 4],
            "bb_norms_atoms": [0.5],  # will encode 3 vectors with this
        }

        self.model = GVP_Network(
            n_layers=depth,
            feats_x_in=48,
            vectors_x_in=7,
            feats_x_out=48,
            vectors_x_out=7,
            feats_edge_in=8,
            vectors_edge_in=1,
            feats_edge_out=8,
            vectors_edge_out=1,
            embedding_nums=[36, 20],
            embedding_dims=[16, 16],
            edge_embedding_nums=[2],
            edge_embedding_dims=[2],
            residual=True,
            recalc=1
        )

        self.noise = noise
        self.init_lr = init_lr

        self.baseline_losses = []
        self.epoch_losses = []

    def forward(self, seq, true_coords, angles, padding_seq, mask):
        needed_info = self.needed_info
        device = true_coords.device

        needed_info["seq"] = seq[: (-padding_seq) or None]
        needed_info["covalent_bond"] = prot_covalent_bond(needed_info["seq"])

        pre_target = encode_whole_protein(
            seq,
            true_coords,
            angles,
            padding_seq,
            needed_info=needed_info,
            free_mem=True,
        )
        pre_target_x, _, _, embedd_info = pre_target

        encoded = encode_whole_protein(
            seq,
            true_coords + self.noise * torch.randn_like(true_coords),
            angles,
            padding_seq,
            needed_info=needed_info,
            free_mem=True,
        )

        x, edge_index, edge_attrs, embedd_info = encoded

        batch = torch.tensor([0 for i in range(x.shape[0])], device=x.device).long()

        # add position coords
        cloud_mask = scn_cloud_mask(seq[: (-padding_seq) or None]).to(device)
        # cloud is all points, chain is all for which we have labels
        chain_mask = mask[: (-padding_seq) or None].unsqueeze(-1) * cloud_mask
        flat_chain_mask = rearrange(chain_mask.bool(), "l c -> (l c)")
        cloud_mask = cloud_mask.bool()
        flat_cloud_mask = rearrange(cloud_mask, "l c -> (l c)")

        recalc_edge = partial(
            encode_whole_bonds,
            x_format="encode",
            embedd_info=embedd_info,
            needed_info=needed_info,
            free_mem=True,
        )

        # predict
        scores = self.model.forward(
            x,
            edge_index,
            batch=batch,
            edge_attr=edge_attrs,
            recalc_edge=recalc_edge,
            verbose=False,
        )

        # format pred, baseline and target
        target = from_encode_to_pred(
            pre_target_x, embedd_info=embedd_info, needed_info=needed_info
        )
        pred = from_encode_to_pred(
            scores, embedd_info=embedd_info, needed_info=needed_info
        )
        base = from_encode_to_pred(x, embedd_info=embedd_info, needed_info=needed_info)

        # MEASURE ERROR

        # option 1: loss is MSE on output tokens
        # loss_ = (target-pred)**2
        # loss  = loss_.mean()

        # option 2: loss is RMSD on reconstructed coords
        target_coords = target[:, 3:4] * target[:, :3]
        pred_coords = pred[:, 3:4] * pred[:, :3]
        base_coords = base[:, 3:4] * base[:, :3]

        ## align - sometimes svc fails - idk why
        try:
            pred_aligned, target_aligned = kabsch_torch(pred_coords.t(), target_coords.t()) # (3, N)
            base_aligned, _ = kabsch_torch(base_coords.t(), target_coords.t())
            loss = ( (pred_aligned.t() - target_aligned.t())[flat_chain_mask[flat_cloud_mask]]**2 ).mean()**0.5 
            loss_base = ( (base_aligned.t() - target_aligned.t())[flat_chain_mask[flat_cloud_mask]]**2 ).mean()**0.5 
        except:
            pred_aligned, target_aligned = None, None
            print("svd failed convergence, ep:", ep)
            loss = ( (pred_coords.t() - target_coords.t())[flat_chain_mask[flat_cloud_mask]]**2 ).mean()**0.5
            loss_base = ( (base_coords - target_coords)[flat_chain_mask[flat_cloud_mask]]**2 ).mean()**0.5 

        # free gpu mem
        del true_coords, angles, pre_target_x, edge_index, edge_attrs
        del scores, target_coords, pred_coords, base_coords
        del encoded, pre_target, target_aligned, pred_aligned
        gc.collect()

        # return loss
        return {"loss": loss, "loss_base": loss_base}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        return optimizer

    def on_train_start(self) -> None:
        self.baseline_losses = []
        self.epoch_losses = []

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss = output["loss"]
        loss_base = output["loss_base"]

        if loss is None or torch.isnan(loss):
            return None

        self.epoch_losses.append(loss.item())
        self.baseline_losses.append(loss_base.item())

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_loss_base", output["loss_base"], on_epoch=True, prog_bar=False)

        return loss

    def on_train_end(self) -> None:
        plt.figure(figsize=(15, 6))
        plt.title(
            f"Loss Evolution - Denoising of Gaussian-masked Coordinates (mu=0, sigma={self.noise})"
        )
        plt.plot(self.epoch_losses, label="train loss step")

        for window in [8, 16, 32]:
            plt.plot(
                [
                    np.mean(self.epoch_losses[:window][0 : i + 1])
                    for i in range(min(window, len(self.epoch_losses)))
                ]
                + [
                    np.mean(self.epoch_losses[i : i + window + 1])
                    for i in range(len(self.epoch_losses) - window)
                ],
                label="Window mean n={0}".format(window),
            )

        plt.plot(
            np.ones(len(self.epoch_losses)) * np.mean(self.baseline_losses),
            "k--",
            label="Baseline",
        )

        plt.xlim(-0.01 * len(self.epoch_losses), 1.01 * len(self.epoch_losses))
        plt.ylabel("RMSD")
        plt.xlabel("Batch number")
        plt.legend()
        plt.savefig("loss.pdf")

    def validation_step(self, batch, batch_idx):
        output = self.forward(**batch)
        self.log("val_loss", output["loss"], on_epoch=True, sync_dist=True)
        self.log("val_loss_base", output["loss_base"], on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        output = self.forward(**batch)
        self.log("test_loss", output["loss"], on_epoch=True, sync_dist=True)
        self.log("test_loss_base", output["loss_base"], on_epoch=True, sync_dist=True)


def get_trainer(args):
    pl.seed_everything(args.seed)

    # loggers
    root_dir = Path(args.default_root_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    tb_save_dir = root_dir / "tb"
    tb_logger = TensorBoardLogger(save_dir=tb_save_dir)
    loggers = [tb_logger]
    logger.info(f"Run tensorboard --logdir {tb_save_dir}")

    # callbacks
    ckpt_cb = ModelCheckpoint(verbose=True)
    lr_cb = LearningRateMonitor(logging_interval="step")
    pb_cb = ProgressBar(refresh_rate=args.progress_bar_refresh_rate)
    callbacks = [lr_cb, pb_cb]

    callbacks.append(ckpt_cb)

    gpu_cb = GPUStatsMonitor()
    callbacks.append(gpu_cb)

    plugins = []
    trainer = pl.Trainer.from_argparse_args(
        args, logger=loggers, callbacks=callbacks, plugins=plugins
    )

    return trainer


def main(args):
    dm = ScnDataModule(**vars(args))
    model = StructureModel(**vars(args))
    trainer = get_trainer(args)
    trainer.fit(model, datamodule=dm)
    metrics = trainer.test(model, datamodule=dm)
    print("test", metrics)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=23333, help="Seed everything.")

    # add model specific args
    parser = StructureModel.add_model_specific_args(parser)

    # add data specific args
    parser = ScnDataModule.add_data_specific_args(parser)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pprint(vars(args))
    main(args)

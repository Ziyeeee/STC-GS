import os
from lightning import Trainer
from argparse import Namespace
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split

from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from models.MambaGRU import PredMamba
from datasets.stcgs_dataset import GaussiansDataset
from configs.MambaGRU import PredMambaConfig


def main(args: Namespace, config: PredMambaConfig):
    model = PredMamba(
        seq_in=config.seq_in,
        seq_out=config.seq_out,
        image_size=(config.image_height, config.image_width),
        view_depth=config.view_depth,
        point_dim=config.point_dim,
        d_model=args.d_model,
        n_layer=args.n_layer,
        d_intermediate=config.d_intermediate,
        ssm_cfg=config.ssm_cfg,
        attn_layer_idx=config.attn_layer_idx,
        attn_cfg=config.attn_cfg,
        rms_norm=config.rms_norm,
        residual_in_fp32=config.residual_in_fp32,
        fused_add_norm=config.fused_add_norm,
        initializer_cfg={"initializer_range": 0.1},
        lr=args.lr,
    )
    train_dataset = GaussiansDataset(args.dataset_dir, args.json_path, mode="train")
    val_dataset = GaussiansDataset(args.dataset_dir, args.json_path, mode="val")
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model.set_norm_denorm(train_dataset)

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=os.path.join(args.save_dir, "checkpoints"),
            save_top_k=1,
            save_last=True,
            filename="model-{epoch:02d}-{val_loss:.4f}",
        )
    ]
    logger = TensorBoardLogger(args.save_dir, name="lightning_logs")
    trainer = Trainer(
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=0.1,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=logger,
        # strategy='ddp_find_unused_parameters_true'
    )

    trainer.fit(model, train_data, val_data)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    seed_everything(0)

    parser = ArgumentParser()
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_layer", type=int, default=32)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--hdf_path", type=str)
    parser.add_argument("--save_dir", type=str, default="outputs")

    args = parser.parse_args()
    main(args, PredMambaConfig())

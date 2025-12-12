import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.R2GenCSR import R2GenCSR
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl

def train(args):
    import torch
    aaa = torch.ones((1,1)).to(f'cuda')
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    # Add checkpoint callback explicitly
    checkpoint_dir = os.path.join(args.savedmodel_path, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epoch={epoch}-step={step}',
        save_top_k=-1,  # Save all checkpoints
        save_last=True,  # Also save a 'last.ckpt'
        every_n_epochs=1,  # Save after every epoch
        verbose=True
    )
    
    # Add to callbacks list
    if "callbacks" not in callbacks or callbacks["callbacks"] is None:
        callbacks["callbacks"] = []
    callbacks["callbacks"].append(checkpoint_callback)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    # Check for existing checkpoint to resume from
    resume_checkpoint = None
    
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoints:
            checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
            resume_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
            print("*" * 50)
            print(f"Found checkpoint to resume from: {resume_checkpoint}")
            print("*" * 50)

    if args.ckpt_file is not None:
        model = R2GenCSR.load_from_checkpoint(args.ckpt_file, args=args, strict=False)
    else:
        model = R2GenCSR(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=resume_checkpoint)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)

if __name__ == '__main__':
    main()

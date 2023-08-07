from typing import Optional

import omegaconf

import hydra
from hydra.utils import _locate, instantiate


def login_wandb(key: Optional[str] = None) -> bool:
    import wandb

    return wandb.login(key=key)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(args: omegaconf.DictConfig) -> None:
    # LightningModule
    config = instantiate(args.runner.model_config)
    model = _locate(args.runner.model_class)(config)
    runner_cls = _locate(args.runner.cls)
    runner = runner_cls(model=model, optimizer=args.runner.optimizer, data=args.data)

    # Pl.Trainer
    if "wandb" in args.trainer.logger._target_:
        login_wandb(args.trainer.wandb_key)
    logger = instantiate(args.trainer.logger)
    callbacks = [instantiate(callback) for callback in args.trainer.callbacks]
    trainer = instantiate(
        args.trainer.cls, logger=logger, callbacks=callbacks
    )
    
    # Training
    trainer.fit(model=runner)


if __name__ == "__main__":
    main()
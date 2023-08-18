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
    loggers = []
    for logger_name, logger in args.trainer.logger.items():
        if "wandb" == logger_name:
            login_wandb(args.wandb_key)
        loggers += [instantiate(logger)]
    callbacks = []
    for callback in args.trainer.callbacks.values():
        callbacks += [instantiate(callback)]
    trainer = instantiate(args.trainer.cls, logger=loggers, callbacks=callbacks)

    # Training
    trainer.fit(model=runner)


if __name__ == "__main__":
    main()

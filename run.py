import omegaconf

import hydra
from hydra.utils import _locate, instantiate


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(args: omegaconf.DictConfig) -> None:
    # LightningModule
    config = instantiate(args.runner.model_config)
    model = _locate(args.runner.model_class)(config)
    runner_cls = _locate(args.runner.cls)
    runner = runner_cls(model=model, optimizer=args.runner.optimizer, data=args.data)

    # Pl.Trainer
    logger = instantiate(args.trainer.logger)
    callbacks = [instantiate(callback) for callback in args.trainer.callbacks]
    trainer = instantiate(args.trainer.cls, callbacks=callbacks)
    
    # Training
    trainer.fit(model=runner)


if __name__ == "__main__":
    main()
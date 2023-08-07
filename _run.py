import hydra
import omegaconf


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(args: omegaconf.DictConfig) -> None:
    config = hydra.utils.instantiate(args.runner.model_config)
    model = hydra.utils._locate(args.runner.model_class)(config)
    runner_cls = hydra.utils._locate(args.runner.cls)
    runner = runner_cls(model=model, optimizer=args.runner.optimizer, data=args.data)
    print("Done")


if __name__ == "__main__":
    main()
from configs.config_scaffold import ModelFramework


def validate_model_framework(config):
    print(config.model.framework)
    try:
        config.model.framework in ModelFramework
    except TypeError:
        raise ValueError(
            f"Model framework {config.model.framework} not recognized. Please choose from {ModelFramework}."
        )
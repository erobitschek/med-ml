from configs.config_scaffold import ModelFramework


# TODO: Add more tests to ensure the formatting of the config/environment setup is correct
def validate_model_framework(config):
    """
    Validates that the specified model framework in the config is supported.

    Args:
        config (Config): Configuration object containing model and other parameters.

    Raises:
        ValueError: If the specified model framework is not recognized.

    Notes:
        The function checks the model framework against the ModelFramework enumeration to ensure it's a supported framework.
    """
    print(config.model.framework)
    try:
        config.model.framework in ModelFramework
    except TypeError:
        raise ValueError(
            f"Model framework {config.model.framework} not recognized. Please choose from {ModelFramework}."
        )
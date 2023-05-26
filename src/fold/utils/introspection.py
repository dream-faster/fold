import inspect


def get_initialization_parameters(instance):
    init_args = inspect.signature(instance.__init__).parameters.keys()
    return {key: value for key, value in instance.__dict__.items() if key in init_args}

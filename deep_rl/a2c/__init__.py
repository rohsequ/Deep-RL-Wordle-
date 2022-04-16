from a2c.sumchars import SumChars, SumCharsLSTM

_registry = {}


def register(ctor, name):
    _registry[name] = ctor


def construct(name, **kwargs):
    return _registry[name](**kwargs)


register(SumChars, "SumChars")
register(SumCharsLSTM, "SumCharsLSTM")

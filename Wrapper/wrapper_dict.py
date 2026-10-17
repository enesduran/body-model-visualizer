"""Registry of body model adapters; each one is optional at import time."""
import importlib
from functools import partial

from loguru import logger

WRAPPER_CLASSES = {}


def _register(module_name, class_name, model_names=None):
    """Import one wrapper; a missing or broken dependency only drops its models.

    ImportError rather than ModuleNotFoundError: an installed but broken
    package (e.g. an mhr/pymomentum ABI mismatch) must skip the model too.
    """
    try:
        wrapper = getattr(importlib.import_module(module_name), class_name)
    except ImportError as error:
        logger.warning(f'{class_name} unavailable, skipping: {error}')
        return
    if model_names is None:
        WRAPPER_CLASSES[class_name.split('_')[0]] = wrapper
    else:
        for name in model_names:
            WRAPPER_CLASSES[name] = partial(wrapper, name)


_register('Wrapper.SMPL_WRAPPER', 'SMPL_WRAPPER', ('SMPL', 'SMPLX', 'MANO', 'FLAME'))
_register('Wrapper.SUPR_WRAPPER', 'SUPR_WRAPPER')
_register('Wrapper.STAR_WRAPPER', 'STAR_WRAPPER')
_register('Wrapper.ANNY_WRAPPER', 'ANNY_WRAPPER')
_register('Wrapper.MHR_WRAPPER', 'MHR_WRAPPER')

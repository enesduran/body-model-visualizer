from loguru import logger

from Wrapper.ANNY_WRAPPER import ANNY_WRAPPER
from Wrapper.STAR_WRAPPER import STAR_WRAPPER
from Wrapper.SUPR_WRAPPER import SUPR_WRAPPER

WRAPPER_CLASSES = {
    'SUPR': SUPR_WRAPPER,
    'ANNY': ANNY_WRAPPER,
    'STAR': STAR_WRAPPER,
}

try:
    from Wrapper.MHR_WRAPPER import MHR_WRAPPER
# ImportError statt nur ModuleNotFoundError: auch ein installiertes, aber
# kaputtes mhr/pymomentum (z.B. ABI-Fehler) soll MHR nur überspringen
except ImportError as e:
    logger.warning(f'MHR unavailable, skipping: {e}')
    MHR_WRAPPER = None
else:
    WRAPPER_CLASSES['MHR'] = MHR_WRAPPER

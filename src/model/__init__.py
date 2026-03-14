from src.model.baseline_model import BaselineModel
from src.model.hifigan import HiFiGAN
from src.model.freev import FreeV
from src.model.discriminators import MultiResolutionDiscriminator,  MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.model.hypothesis_1 import FreeVH1

__all__ = [
    "BaselineModel",
]

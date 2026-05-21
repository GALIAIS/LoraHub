# LoRA module building blocks. Public API re-exported here so
# `from networks.lora_modules import LoRAModule, ...` works unchanged.

from networks.lora_modules.base import BaseLoRAModule, _absorb_channel_scale
from networks.lora_modules.boft import BOFTModule
from networks.lora_modules.chimera import (
    ChimeraHydraInferenceModule,
    ChimeraHydraLoRAModule,
)
from networks.lora_modules.diag_oft import DiagOFTModule
from networks.lora_modules.dora import DoRAModule
from networks.lora_modules.dylora import DyLoRAModule
from networks.lora_modules.full import FullModule
from networks.lora_modules.hydra import HydraLoRAModule, _sigma_sinusoidal_features
from networks.lora_modules.ia3 import IA3Module
from networks.lora_modules.loha import LoHAModule
from networks.lora_modules.lokr import LoKrModule
from networks.lora_modules.lora import LoRAModule
from networks.lora_modules.ortho import (
    OrthoHydraLoRAModule,
    OrthoLoRAModule,
)
from networks.lora_modules.reft import ReFTModule
from networks.lora_modules.stacked_experts import StackedExpertsLoRAModule

__all__ = [
    "BaseLoRAModule",
    "BOFTModule",
    "ChimeraHydraInferenceModule",
    "ChimeraHydraLoRAModule",
    "DiagOFTModule",
    "DoRAModule",
    "DyLoRAModule",
    "FullModule",
    "HydraLoRAModule",
    "IA3Module",
    "LoHAModule",
    "LoKrModule",
    "LoRAModule",
    "OrthoHydraLoRAModule",
    "OrthoLoRAModule",
    "ReFTModule",
    "StackedExpertsLoRAModule",
    "_absorb_channel_scale",
    "_sigma_sinusoidal_features",
]

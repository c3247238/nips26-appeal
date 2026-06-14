"""Dynamic weight decay optimizer implementations."""

from .base import WDOptimizer
from .udwdc import UDWDCOptimizer
from .udwdc_v2 import UDWDCv2Optimizer
from .cwd import CWDOptimizer
from .swd import SWDOptimizer
from .cpr import CPROptimizer
from .defazio import DefazioCorrective
from .fixed_wd import FixedWDOptimizer
from .no_wd import NoWDOptimizer

METHOD_REGISTRY = {
    "FixedWD": FixedWDOptimizer,
    "CWD": CWDOptimizer,
    "SWD": SWDOptimizer,
    "CPR": CPROptimizer,
    "DefazioCorrective": DefazioCorrective,
    "NoWD": NoWDOptimizer,
    "UDWDC": UDWDCOptimizer,
    "UDWDC-v2": UDWDCv2Optimizer,
}

def create_optimizer(method_name: str, model, lr: float, momentum: float = 0.9,
                     weight_decay: float = 1e-4, **kwargs):
    """Factory function to create an optimizer by method name."""
    if method_name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHOD_REGISTRY.keys())}")
    cls = METHOD_REGISTRY[method_name]
    return cls(model, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)

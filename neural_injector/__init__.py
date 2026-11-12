__version__ = "0.7.0"

from .analyzer import (
    DEFAULT_BAND_EDGES,
    CheckpointEvolutionTracker,
    ContrastivePoleFinder,
    LogitLensAttributor,
    StreamingLayerStats,
)
from .circuit_tracer import CircuitTracer
from .discovery import ARCHITECTURE_REGISTRY, ArchitectureAdapter, LayerDescriptor, LayerDiscovery
from .explainer import (
    PAPER_STRONG_FRACTION,
    CallableLLMClient,
    CircuitEcologyMap,
    LLMClientProtocol,
    NeuronExplainerAgent,
    OpenAICompatibleClient,
)
from .exporter import ProductionSteeringRuntime, SteeringExporter
from .hooks import ActivationSite, HookSet, resolve_site
from .injector import NeuralInjector
from .interventions import AblationMaskOptimizer, InterventionKernel, build_intervention_hookset
from .session import WorkbenchSession
from .types import *  # noqa: F401,F403
from . import types as _types

__all__ = [
    "__version__",
    "DEFAULT_BAND_EDGES",
    "PAPER_STRONG_FRACTION",
    "ARCHITECTURE_REGISTRY",
    "ArchitectureAdapter",
    "LayerDescriptor",
    "LayerDiscovery",
    "StreamingLayerStats",
    "LogitLensAttributor",
    "ContrastivePoleFinder",
    "CheckpointEvolutionTracker",
    "CircuitTracer",
    "CallableLLMClient",
    "CircuitEcologyMap",
    "LLMClientProtocol",
    "NeuronExplainerAgent",
    "OpenAICompatibleClient",
    "ProductionSteeringRuntime",
    "SteeringExporter",
    "ActivationSite",
    "HookSet",
    "resolve_site",
    "NeuralInjector",
    "AblationMaskOptimizer",
    "InterventionKernel",
    "build_intervention_hookset",
    "WorkbenchSession",
] + [k for k, v in vars(_types).items() if isinstance(v, type) and v.__module__.endswith(".types")]

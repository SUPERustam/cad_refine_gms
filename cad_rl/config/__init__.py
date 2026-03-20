"""Typed configuration schema and profile resolver."""

from .profile import ProfileResolver, load_profile_document, resolve_experiment_profile
from .schema import (
    AlgorithmSpec,
    CheckpointRef,
    ComparisonReport,
    EvalRecord,
    InferenceRecord,
    MachineProfile,
    MeshRecord,
    ModelSpec,
    ResolvedExperimentConfig,
    RunManifest,
    TaskSpec,
    TrainerSpec,
    to_serializable,
)

__all__ = [
    "AlgorithmSpec",
    "CheckpointRef",
    "ComparisonReport",
    "EvalRecord",
    "InferenceRecord",
    "MachineProfile",
    "MeshRecord",
    "ModelSpec",
    "ProfileResolver",
    "ResolvedExperimentConfig",
    "RunManifest",
    "TaskSpec",
    "TrainerSpec",
    "load_profile_document",
    "resolve_experiment_profile",
    "to_serializable",
]

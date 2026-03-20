from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency
    from comet_ml import Experiment as CometExperiment  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CometExperiment = None


@dataclass(slots=True)
class CometAdapter:
    experiment: Any | None = None
    enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def disabled(cls) -> "CometAdapter":
        return cls(experiment=None, enabled=False)

    @classmethod
    def from_experiment(cls, experiment: Any, **metadata: Any) -> "CometAdapter":
        return cls(experiment=experiment, enabled=True, metadata=dict(metadata))

    @classmethod
    def create_if_available(
        cls,
        *,
        enabled: bool = False,
        project_name: str | None = None,
        workspace: str | None = None,
        **kwargs: Any,
    ) -> "CometAdapter":
        if not enabled or CometExperiment is None:
            return cls.disabled()
        experiment = CometExperiment(
            project_name=project_name, workspace=workspace, **kwargs
        )
        return cls.from_experiment(
            experiment, project_name=project_name, workspace=workspace
        )

    def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        experiment = self.experiment
        if experiment is None:
            return None
        method = getattr(experiment, method_name, None)
        if method is None:
            return None
        return method(*args, **kwargs)

    def log_config(self, config: Mapping[str, Any]) -> None:
        self._call("log_parameters", dict(config))

    def log_metrics(
        self, metrics: Mapping[str, Any], *, step: int | None = None
    ) -> None:
        payload = dict(metrics)
        if step is not None:
            self._call("log_metrics", payload, step=step)
            return
        self._call("log_metrics", payload)

    def log_text(self, name: str, text: str) -> None:
        self._call("log_text", text, metadata={"name": name})

    def log_asset(self, file_path: str, **kwargs: Any) -> None:
        self._call("log_asset", file_path, **kwargs)

    def log_table(self, name: str, table: Any) -> None:
        self._call("log_table", name, table)

    def log_artifact(self, file_path: str, **kwargs: Any) -> None:
        self._call("log_asset", file_path, **kwargs)

    def set_metadata(self, **metadata: Any) -> None:
        self.metadata.update(metadata)
        self._call("set_context", self.metadata)

    def finish(self) -> None:
        self._call("end")

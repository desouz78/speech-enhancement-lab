"""Explainable, capability-based AI model selection.

Selection matches an explicitly requested processing task against the model
registry. It does not infer the task from audio, rank model quality, execute
models, or predict deployment feasibility from acoustic features.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from audio_lab.models import (
    AudioTask,
    BrowserStatus,
    IntegrationStatus,
    MODEL_CATALOG,
    ModelSpec,
)


class SelectionTier(StrEnum):
    """The minimum declared availability required of each candidate."""

    CATALOG = "catalog"
    INTEGRATED = "integrated"
    BROWSER = "browser"


class SelectionReason(StrEnum):
    """Stable reason identifiers for UI rendering and audit logs."""

    TASK_SUPPORTED = "task_supported"
    ADAPTER_NOT_INTEGRATED = "adapter_not_integrated"
    BROWSER_NOT_ASSESSED = "browser_not_assessed"
    BROWSER_NOT_FEASIBLE = "browser_not_feasible"


@dataclass(frozen=True, slots=True)
class ModelAssessment:
    """One matching model and its eligibility under the requested tier.

    Eligibility at the CATALOG tier means capability matching only; it does
    not imply successful inference, verified quality, or deployment readiness.
    """

    model: ModelSpec
    eligible: bool
    reasons: tuple[SelectionReason, ...]


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Immutable selection outcome in registry order, without model ranking."""

    task: AudioTask
    tier: SelectionTier
    assessments: tuple[ModelAssessment, ...]

    @property
    def eligible_models(self) -> tuple[ModelSpec, ...]:
        """Models meeting the declared requirements of this selection tier."""
        return tuple(item.model for item in self.assessments if item.eligible)

    @property
    def excluded_models(self) -> tuple[ModelSpec, ...]:
        """Models with the requested capability but insufficient status."""
        return tuple(item.model for item in self.assessments if not item.eligible)


def _parse_task(task: AudioTask | str) -> AudioTask:
    if isinstance(task, AudioTask):
        return task
    if not isinstance(task, str):
        raise TypeError("task must be an AudioTask or its exact string value.")
    try:
        return AudioTask(task)
    except ValueError as exc:
        raise ValueError("Unknown audio task.") from exc


def _parse_tier(tier: SelectionTier | str) -> SelectionTier:
    if isinstance(tier, SelectionTier):
        return tier
    if not isinstance(tier, str):
        raise TypeError("tier must be a SelectionTier or its exact string value.")
    try:
        return SelectionTier(tier)
    except ValueError as exc:
        raise ValueError("Unknown selection tier.") from exc


def _assess(model: ModelSpec, tier: SelectionTier) -> ModelAssessment:
    """Assess declared status; never infer runtime behavior or model quality."""
    reasons = [SelectionReason.TASK_SUPPORTED]
    integrated = model.integration_status is IntegrationStatus.INTEGRATED
    browser_feasible = model.browser_status is BrowserStatus.FEASIBLE

    if not integrated:
        reasons.append(SelectionReason.ADAPTER_NOT_INTEGRATED)
    if model.browser_status is BrowserStatus.NOT_ASSESSED:
        reasons.append(SelectionReason.BROWSER_NOT_ASSESSED)
    elif model.browser_status is BrowserStatus.NOT_FEASIBLE:
        reasons.append(SelectionReason.BROWSER_NOT_FEASIBLE)

    eligible = (
        tier is SelectionTier.CATALOG
        or (tier is SelectionTier.INTEGRATED and integrated)
        or (tier is SelectionTier.BROWSER and integrated and browser_feasible)
    )
    return ModelAssessment(model, eligible, tuple(reasons))


def select_models(
    task: AudioTask | str,
    *,
    tier: SelectionTier | str = SelectionTier.CATALOG,
) -> SelectionResult:
    """Find candidates for one explicitly specified audio-processing task.

    CATALOG: return declared-capability matches, even when not integrated.
    INTEGRATED: require an integrated adapter, not runtime verification.
    BROWSER: require integration and an affirmative browser-feasibility status.

    The catalog is not ranked. A zero-candidate result is valid and must not
    be presented as an available processing route by a consuming interface.
    """
    parsed_task = _parse_task(task)
    parsed_tier = _parse_tier(tier)
    matching = (model for model in MODEL_CATALOG if parsed_task in model.tasks)
    assessments = tuple(_assess(model, parsed_tier) for model in matching)
    return SelectionResult(parsed_task, parsed_tier, assessments)

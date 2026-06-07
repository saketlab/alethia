"""Label-free embedding-model assessment for entity matching."""

from .assessor import (
    AssessmentReport,
    ModelAssessment,
    assess_models,
    assessment_table,
)
from .simulate import generate_positive_pairs, make_dirty_variant
from .validate import LabeledDataset, ValidationResult, true_accuracy, validate_assessor

__all__ = [
    "assess_models",
    "assessment_table",
    "AssessmentReport",
    "ModelAssessment",
    "generate_positive_pairs",
    "make_dirty_variant",
    "validate_assessor",
    "LabeledDataset",
    "ValidationResult",
    "true_accuracy",
]

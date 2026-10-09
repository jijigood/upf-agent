"""Versioned, source-bound knowledge annotations shared by retrieval and graph writeback."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from upf_agent.v8.models import EvidenceBinding, Model, SourcePointer, digest


class KnowledgeAnnotation(Model):
    annotation_id: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    product: str
    version: str
    object_ids: list[str] = Field(min_length=1)
    purposes: list[
        Literal["upf_n4_fault", "upf_n3_fault", "manual_operation", "log_observation"]
    ] = Field(min_length=1)
    basis_kind: Literal["expert_review", "trusted_rule"]
    basis_source: str = Field(min_length=1)
    basis_version: str = Field(min_length=1)
    reviewed: bool = False
    reviewer: str | None = None
    rule_id: str | None = None
    revoked: bool = False

    @model_validator(mode="after")
    def basis(self):
        if self.reviewed and not self.reviewer:
            raise ValueError("reviewed annotations require a reviewer")
        if self.basis_kind == "trusted_rule" and not self.rule_id:
            raise ValueError("trusted rule annotations require a rule ID")
        return self

    @property
    def usable(self):
        return not self.revoked and (self.basis_kind == "trusted_rule" or self.reviewed)


class KnowledgeRegistry(Model):
    schema_version: Literal["v8-knowledge-applicability-1"] = "v8-knowledge-applicability-1"
    version: str = Field(min_length=1)
    annotations: list[KnowledgeAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique(self):
        ids = [item.annotation_id for item in self.annotations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate annotation IDs")
        return self

    @classmethod
    def load(cls, root: Path):
        path = root / "data/v8/knowledge_applicability.json"
        return cls.model_validate_json(path.read_text()) if path.exists() else cls(version="absent")

    def resolve(self, source: SourcePointer, text: str, product, version, object_id):
        content_sha = (
            hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) and text else None
        )
        found = [
            item
            for item in self.annotations
            if (
                content_sha is not None
                and item.source_sha256 == source.sha256
                and item.locator == source.locator
                and item.content_sha256 == content_sha
                and item.product == product
                and item.version == version
                and object_id in item.object_ids
                and item.usable
            )
        ]
        purpose_sets = {tuple(sorted(item.purposes)) for item in found}
        status = (
            "invalid_content"
            if content_sha is None
            else "mapped"
            if len(purpose_sets) == 1
            else "conflicting"
            if found
            else "unknown"
        )
        return {
            "registry_version": self.version,
            "registry_sha256": digest(self),
            "status": status,
            "annotation_ids": sorted(item.annotation_id for item in found),
            "purposes": list(next(iter(purpose_sets))) if status == "mapped" else [],
            "reviewed": bool(found) and all(item.reviewed for item in found),
        }


def annotate_knowledge(root: Path, binding: EvidenceBinding, registry=None):
    from upf_agent.v8.adapters import verify_binding

    if binding.category != "K":
        raise ValueError("knowledge annotation requires K evidence")
    registry = registry or KnowledgeRegistry.load(root)
    receipt = registry.resolve(
        binding.source,
        binding.fields.get("instruction", ""),
        binding.product,
        binding.version,
        binding.object_id,
    )
    binding.source_verified = verify_binding(root, binding)
    if not binding.source_verified:
        receipt["status"] = "source_unverified"
        receipt["purposes"] = []
    binding.purposes = receipt["purposes"]
    binding.knowledge_applicability = receipt
    return binding

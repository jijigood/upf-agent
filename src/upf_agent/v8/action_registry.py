"""Finite parameter schemas; no arbitrary commands or model predicate execution."""

from typing import Literal

from pydantic import model_validator

from upf_agent.v8.models import Model
from upf_agent.v8.plan_ir import REGISTRY_VERSION


class EmptyParams(Model):
    pass


class InterfaceParams(Model):
    interface: Literal["N3", "N4", "N6", "N9"]


class ConfigurationParams(InterfaceParams):
    field: Literal["n4_enabled", "mtu"]


class PeerParams(Model):
    peer_object_id: str


class LogParams(Model):
    event_type: str | None = None


class EvidenceParams(Model):
    missing: list[str]


class ChangeParams(Model):
    field: Literal["n4_enabled", "mtu"]
    proposed_value: bool | int
    requires_approval: Literal[True]

    @model_validator(mode="after")
    def bounds(self):
        if self.field == "n4_enabled" and type(self.proposed_value) is not bool:
            raise ValueError("n4_enabled must be boolean")
        if self.field == "mtu" and (
            type(self.proposed_value) is not int or not 576 <= self.proposed_value <= 9216
        ):
            raise ValueError("mtu outside bounded proposal range")
        return self


# action -> schema, permission, effect, readable fields, necessary known K
ACTIONS = {
    "CHECK_IDENTITY": (EmptyParams, "read", "read", [], False),
    "CHECK_INTERFACE": (InterfaceParams, "read", "read", ["interface", "state"], False),
    "CHECK_CONFIGURATION": (ConfigurationParams, "read", "read", [], True),
    "READ_LOG": (LogParams, "read", "read", ["event_type"], False),
    "VERIFY_PEER_MAPPING": (PeerParams, "read", "read", ["mapping_version", "service_id"], False),
    "REQUEST_EVIDENCE": (EvidenceParams, "human", "suggestion", [], False),
    "HUMAN_REVIEW": (EmptyParams, "human", "suggestion", [], False),
    "PROPOSE_CHANGE": (ChangeParams, "suggest", "suggestion", [], True),
}


def registry_manifest():
    return {
        "version": REGISTRY_VERSION,
        "actions": {
            name: {
                "parameters": item[0].model_json_schema(),
                "permission_kind": item[1],
                "effect_kind": item[2],
                "readable_fields": item[3],
                "requires_knowledge": item[4],
            }
            for name, item in ACTIONS.items()
        },
    }

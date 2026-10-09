"""Lossless bounded JSON translator. No repair, enrichment or model evaluation calls."""

import hashlib
import json

from pydantic import ValidationError

from upf_agent.v8.plan_ir import PlanIR, StepSpan, TranslationReceipt


def _object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def translate_plan(raw: str):
    receipt = TranslationReceipt(raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
    if len(raw) > 100_000:
        receipt.errors.append("raw_plan_too_large")
        return receipt
    decoder = json.JSONDecoder(
        object_pairs_hook=_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    try:
        value = decoder.decode(raw)
        receipt.plan = PlanIR.model_validate(value)
        # Locate original object spans by scanning JSON values rather than substring guesses.
        for pos, char in enumerate(raw):
            if char != "{":
                continue
            try:
                item, end = decoder.raw_decode(raw, pos)
            except ValueError:
                continue
            if not isinstance(item, dict) or "step_id" not in item or "text" not in item:
                continue
            # Locate the original encoded text value within this step object.
            cursor = pos + 1
            text_start = text_end = None
            while cursor < end:
                while cursor < end and raw[cursor] in " \t\r\n,":
                    cursor += 1
                if cursor >= end or raw[cursor] == "}":
                    break
                key, cursor = decoder.raw_decode(raw, cursor)
                while raw[cursor] in " \t\r\n":
                    cursor += 1
                if raw[cursor] != ":":
                    raise ValueError("invalid field separator")
                cursor += 1
                while raw[cursor] in " \t\r\n":
                    cursor += 1
                start = cursor
                _, cursor = decoder.raw_decode(raw, cursor)
                if key == "text":
                    text_start, text_end = start, cursor
            if text_start is not None:
                receipt.spans.append(
                    StepSpan(
                        step_id=item["step_id"],
                        start=pos,
                        end=end,
                        text_start=text_start,
                        text_end=text_end,
                    )
                )
        ids = [span.step_id for span in receipt.spans]
        if ids != [step.step_id for step in receipt.plan.steps]:
            receipt.plan = None
            receipt.errors.append("ambiguous_step_spans")
    except ValidationError as error:
        receipt.errors = [f"{'.'.join(map(str, e['loc']))}:{e['type']}" for e in error.errors()]
        receipt.unknown_fields = [
            str(e["loc"]) for e in error.errors() if e["type"] == "extra_forbidden"
        ]
    except (ValueError, TypeError, IndexError) as error:
        receipt.plan = None
        receipt.errors.append(f"parse_error:{error}")
    return receipt

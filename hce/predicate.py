"""The flat predicate grammar that decides which tasks a change can affect.

Deliberately small. Keys are ANDed, values are a literal or one comparison, and
there is no nesting. Two reasons beyond simplicity:

  * Every key must exist in FEATURE_SCHEMA, checked when the contract loads.
    That is the main defence against a change conditioned on something the
    engine cannot evaluate -- the agent can only talk about behaviour that was
    actually measured.
  * A predicate that cannot express "or" cannot quietly widen itself. A change
    that really has two triggers has to be declared as two changes, each with
    its own activation set, which is also what the attribution report needs.

A task missing the feature *matches*. Skipping a task because nothing is known
about it is the one error mode with no recovery: it would be silently dropped
from both the improvement and the regression buckets.
"""

from __future__ import annotations

import re

_COMPARISON = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")
_OPS = {
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


class PredicateError(ValueError):
    """A predicate that cannot be evaluated. Raised at contract-load time."""


def validate(predicate: dict, schema: dict) -> None:
    """Raise PredicateError unless every key and value is evaluable."""
    if not isinstance(predicate, dict):
        raise PredicateError("activation_predicate must be an object")
    if not predicate:
        raise PredicateError(
            "activation_predicate is empty; a change with no trigger condition "
            "cannot be verified on a subset")

    for key, cond in predicate.items():
        if key not in schema:
            legal = ", ".join(sorted(schema))
            raise PredicateError(f"unknown feature {key!r}; legal keys are: {legal}")
        kind = schema[key]["type"]
        if isinstance(cond, bool):
            if kind != "bool":
                raise PredicateError(
                    f"{key!r} is {kind}; a boolean literal cannot test it -- "
                    f"use a comparison such as \">1000\"")
            continue
        if isinstance(cond, (int, float)):
            if kind == "bool":
                raise PredicateError(f"{key!r} is boolean; use true or false")
            continue
        if isinstance(cond, str):
            if cond.strip() == "*":
                continue
            if not _COMPARISON.match(cond):
                raise PredicateError(
                    f"{key!r}: cannot parse {cond!r}; expected true/false, a number, "
                    f"or a comparison like \">8000\"")
            if kind == "bool":
                raise PredicateError(f"{key!r} is boolean; use true or false")
            continue
        raise PredicateError(f"{key!r}: unsupported condition type {type(cond).__name__}")


def _test(cond, value) -> bool:
    if isinstance(cond, str) and cond.strip() == "*":
        return True
    if isinstance(cond, bool):
        return bool(value) is cond
    if isinstance(cond, (int, float)):
        return value == cond
    op, number = _COMPARISON.match(cond).groups()
    return _OPS[op](value, float(number))


def matches(predicate: dict, mechanism: dict) -> tuple[bool, list[str]]:
    """Evaluate one task's mechanism profile. Returns (matched, unknown_keys)."""
    unknown = []
    for key, cond in predicate.items():
        value = mechanism.get(key)
        if value is None:
            # Never measured for this task: match, and say so, so the caller can
            # report how much of the activation set rests on missing data.
            unknown.append(key)
            continue
        if not _test(cond, value):
            return False, unknown
    return True, unknown


def activation_set(predicate: dict, mechanisms: dict[str, dict]) -> tuple[list[str], dict[str, list[str]]]:
    """Tasks whose profile satisfies the predicate, plus their unknown keys."""
    selected, unknown = [], {}
    for task in sorted(mechanisms):
        matched, missing = matches(predicate, mechanisms[task])
        if matched:
            selected.append(task)
            if missing:
                unknown[task] = missing
    return selected, unknown

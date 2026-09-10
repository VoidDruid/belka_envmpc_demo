"""Safe parsing helpers for numeric values stored in JSON configs."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any

from simpleeval import (
    FeatureNotAvailable,
    FunctionNotDefined,
    InvalidExpression,
    NameNotDefined,
    OperatorNotDefined,
    SimpleEval,
    safe_power,
)


_NUMERIC_NAMES = {
    "pi": math.pi,
    "tau": math.tau,
    "e": math.e,
    "deg": math.pi / 180.0,
    "inf": math.inf,
}

_NUMERIC_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: safe_power,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def require_exact_keys(
    mapping: Mapping[str, Any], expected: Iterable[str], *, field_name: str
) -> None:
    """Require one JSON object to contain exactly the declared fields."""
    expected_keys = set(expected)
    actual_keys = set(mapping)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ValueError(f"{field_name} has invalid fields: {'; '.join(details)}")


def parse_bool(value: Any, *, field_name: str = "value") -> bool:
    """Parse a JSON boolean without accepting truthy strings or numbers."""
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be boolean, got {type(value).__name__}")
    return value


def parse_numeric(value: Any, *, field_name: str = "value") -> float:
    """Parse one JSON numeric value or a string arithmetic expression into float."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must be finite, got {number!r}")
        return number
    if isinstance(value, str):
        evaluator = SimpleEval(
            names=_NUMERIC_NAMES,
            functions={},
            operators=_NUMERIC_OPERATORS,
            allowed_attrs={},
        )
        try:
            result = evaluator.eval(value)
        except (
            FeatureNotAvailable,
            FunctionNotDefined,
            InvalidExpression,
            KeyError,
            NameNotDefined,
            OperatorNotDefined,
            SyntaxError,
        ) as exc:
            raise ValueError(
                f"{field_name} contains unsupported numeric expression: {value!r}"
            ) from exc
        if isinstance(result, bool) or not isinstance(result, Real):
            raise TypeError(
                f"{field_name} expression must evaluate to a number: {value!r}"
            )
        number = float(result)
        if not math.isfinite(number):
            raise ValueError(f"{field_name} must be finite, got {number!r}")
        return number
    raise TypeError(
        f"{field_name} must be numeric or a numeric expression string, got {type(value).__name__}"
    )


def parse_int(value: Any, *, field_name: str = "value") -> int:
    """Parse one JSON integer value or expression and reject fractional results."""
    number = parse_numeric(value, field_name=field_name)  # parsed numeric scalar
    if not math.isfinite(number) or not float(number).is_integer():
        raise ValueError(
            f"{field_name} must evaluate to a finite integer, got {number!r}"
        )
    return int(number)


def parse_numeric_vector(value: Any, *, field_name: str = "value") -> list[float]:
    """Parse a JSON numeric vector, allowing expressions in individual elements."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be a numeric sequence")
    return [
        parse_numeric(item, field_name=f"{field_name}[{idx}]")
        for idx, item in enumerate(value)
    ]


def parse_int_group(value: Any, *, field_name: str = "value") -> tuple[int, ...]:
    """Parse a JSON sequence of integer channel indexes."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise TypeError(f"{field_name} must be an integer sequence")
    indexes = []  # parsed channel indexes
    for idx, item in enumerate(value):
        if isinstance(item, Integral) and not isinstance(item, bool):
            indexes.append(int(item))
        else:
            indexes.append(parse_int(item, field_name=f"{field_name}[{idx}]"))
    return tuple(indexes)

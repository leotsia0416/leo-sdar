from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, Union, get_args, get_origin, get_type_hints

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - import stays lazy-friendly.
    yaml = None

T = TypeVar("T")


class SerializableMixin:
    """Small dataclass mixin for stable dict/json serialization."""

    def to_dict(self) -> dict[str, Any]:
        return to_serializable(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=True)

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
        return from_mapping(cls, data)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if is_dataclass(obj):
        return {
            field.name: to_serializable(getattr(obj, field.name))
            for field in fields(obj)
        }

    if isinstance(obj, Mapping):
        return {str(key): to_serializable(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(value) for value in obj]

    return obj


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_module = _require_yaml()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml_module.safe_load(handle) or {}

    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(payload).__name__}.")

    return dict(payload)


def dump_yaml_file(path: str | Path, data: Mapping[str, Any]) -> None:
    yaml_module = _require_yaml()
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml_module.safe_dump(
            to_serializable(dict(data)),
            handle,
            sort_keys=False,
            allow_unicode=False,
        )


def from_mapping(cls: type[T], payload: Mapping[str, Any]) -> T:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected mapping data for {cls.__name__}, got {type(payload).__name__}.")

    field_names = {field.name for field in fields(cls)}
    extra_keys = sorted(set(payload) - field_names)
    if extra_keys:
        raise ValueError(f"Unknown fields for {cls.__name__}: {', '.join(extra_keys)}")

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in payload:
            continue
        annotation = type_hints.get(field.name, field.type)
        kwargs[field.name] = _coerce_value(annotation, payload[field.name])

    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    if annotation in (Any, object):
        return value

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        union_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(union_args) == 1:
            return _coerce_value(union_args[0], value)
        for candidate in union_args:
            try:
                return _coerce_value(candidate, value)
            except (TypeError, ValueError):
                continue
        return value

    if origin in (list, set):
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        sequence = [_coerce_value(item_type, item) for item in value]
        return sequence if origin is list else set(sequence)

    if origin is tuple:
        tuple_args = get_args(annotation)
        if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
            return tuple(_coerce_value(tuple_args[0], item) for item in value)
        return tuple(_coerce_value(arg, item) for arg, item in zip(tuple_args, value))

    if origin is dict:
        key_type, value_type = get_args(annotation) if get_args(annotation) else (Any, Any)
        return {
            _coerce_value(key_type, key): _coerce_value(value_type, item)
            for key, item in value.items()
        }

    if annotation is Path:
        return Path(value)

    if isinstance(annotation, type):
        if is_dataclass(annotation) and isinstance(value, Mapping):
            return from_mapping(annotation, value)

        if annotation is bool:
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    return True
                if normalized in {"false", "0", "no", "n", "off"}:
                    return False
                raise ValueError(f"Cannot coerce '{value}' to bool.")
            return bool(value)

        if annotation in (int, float, str):
            return annotation(value)

    return value


def _require_yaml():
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for remask policy config loading. Install pyyaml to use YAML-backed configs."
        )
    return yaml

"""Shared pydantic ``model_config`` for the schema submodules.

Every YAML field is accepted both in its Python snake_case form and in
camelCase (the canonical wire form going forward). ``populate_by_name=True``
keeps existing configs valid; ``extra="forbid"`` is applied per-model where
appropriate.
"""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic.alias_generators import to_camel


_CAMEL_CONFIG = ConfigDict(alias_generator=to_camel, populate_by_name=True)

# Zero-Dependency Core

## Rule
`pdfsys-core` MUST import only from the Python standard library, or from `pdfsys-types` (the canonical type-definition layer vendored in the parsers submodule).

## DO

```python
# Good: stdlib-only imports in pdfsys-core
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
import json
import os

from pdfsys_types.extract import ExtractedDoc  # OK — pdfsys-types is itself zero-dep
```

## DON'T

```python
# Bad: external dependency in pdfsys-core
import numpy as np          # NO — numpy is external
from PIL import Image       # NO — Pillow is external
import pymupdf              # NO — pymupdf is external
```

## Why
Every other package imports `pdfsys-core`. If core pulls in torch or pymupdf, a user who only needs the data types gets a 2 GB dependency chain. Core stays lightweight so downstream packages choose their own deps. pdfsys-types is itself zero-dep by the same rule (no stdlib-external imports), enforced by the same boundary test, so allowing pdfsys-core to depend on it preserves the no-fat-dep-chain guarantee.

## Exceptions
Only `pdfsys-types`, which is itself zero-dep (also enforced by `tests/architecture/test_boundary.py`). No other exceptions. Adding `numpy`, `torch`, `pymupdf`, etc. to either pdfsys-core or pdfsys-types is forbidden and breaks the boundary test.

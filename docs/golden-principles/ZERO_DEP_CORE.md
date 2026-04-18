# Zero-Dependency Core

## Rule
`pdfsys-core` MUST import only from the Python standard library.

## DO

```python
# Good: stdlib-only imports in pdfsys-core
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
import json
import os
```

## DON'T

```python
# Bad: external dependency in pdfsys-core
import numpy as np          # NO — numpy is external
from PIL import Image       # NO — Pillow is external
import pymupdf              # NO — pymupdf is external
```

## Why
Every other package imports `pdfsys-core`. If core pulls in torch or pymupdf, a user who only needs the data types gets a 2 GB dependency chain. Core stays lightweight so downstream packages choose their own deps.

## Exceptions
None. This rule is absolute and enforced by `tests/architecture/test_boundary.py`.

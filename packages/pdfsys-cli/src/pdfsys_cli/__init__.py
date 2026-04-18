"""pdfsys-cli — unified CLI for the pdfsys PDF processing pipeline.

Run individual stages or chain them via YAML config::

    pdfsys init-config > pdfsys.yaml
    pdfsys run -c pdfsys.yaml --stages router,layout,extract
"""

__version__ = "0.0.1"

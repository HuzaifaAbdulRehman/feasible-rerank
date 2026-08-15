"""qubo-rerank: fair, energy-aware recommendation list selection via QUBO."""

from .problem import RerankInstance

__all__ = ["RerankInstance", "__version__"]
# Kept in step with pyproject.toml by tests/test_packaging.py -- a version that
# disagrees with the package metadata is the kind of thing nobody notices until
# it is quoted in a citation.
__version__ = "0.2.0"

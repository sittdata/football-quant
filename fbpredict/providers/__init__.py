"""Data providers.

Every provider returns rows shaped like the ``matches`` table and archives its
raw response, so the report can point at the exact bytes that were fetched.
"""

from .base import BaseProvider, ProviderError, Response

__all__ = ["BaseProvider", "ProviderError", "Response"]

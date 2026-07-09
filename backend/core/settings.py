from __future__ import annotations

# Re-export for backwards compatibility.
# Any module doing `from backend.core.settings import get_settings` will work
# exactly the same as `from backend.core.config import get_settings`.
from backend.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]

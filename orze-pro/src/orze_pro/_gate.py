"""License gate for orze-pro modules."""
from orze_pro.license import is_licensed

def require_license():
    """Raise ImportError if not licensed. Call at top of each pro module."""
    if not is_licensed():
        raise ImportError(
            "orze-pro is not activated. Set ORZE_PRO_KEY environment variable. "
            "Get a license at ANON.example/pro"
        )

"""orze-pro — autopilot for GPU experiments."""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("orze-pro")
except PackageNotFoundError:
    __version__ = "unknown"

from orze_pro.license import is_licensed, license_info, check_license

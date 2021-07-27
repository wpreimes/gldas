import os
import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Checks if .gldasrc existis in /home
# if not, it assumes that the env variable was set
# in a different way and already exists.
from pathlib import Path
home = str(Path.home())
gldasrc = os.path.join(home, '.gldasrc')
if os.path.isfile(gldasrc):
    from dotenv import load_dotenv
    load_dotenv(gldasrc)

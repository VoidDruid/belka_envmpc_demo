import os
import site
from pkgutil import extend_path


for pycyphal_path in os.environ.get("PYCYPHAL_PATH", "").split(os.pathsep):
    if pycyphal_path:
        site.addsitedir(pycyphal_path)


__path__ = extend_path(__path__, __name__)

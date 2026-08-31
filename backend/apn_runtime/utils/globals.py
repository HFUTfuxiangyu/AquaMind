"""Runtime logger expected by the upstream APN implementation.

Training-only accelerator and experiment initialization are intentionally not
included in the standalone AquaMind source package.
"""

import logging


logger = logging.getLogger("AquaMind.APN")

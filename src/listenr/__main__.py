"""Allow ``python -m listenr`` to behave like the ``listenr`` console script."""

import sys

from listenr.main import main

sys.exit(main())

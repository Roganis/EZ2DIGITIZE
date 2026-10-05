# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Entry point of the packaged app: the GUI, or the CLI when given a command.

EZ2DIGITIZE.AppImage                    the GUI
EZ2DIGITIZE.AppImage run ~/scans/skull  the `ez2d` CLI, headless
"""

import sys

from ez2digitize import cli

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in (*cli.commands(), "-h", "--help"):
        sys.exit(cli.main(sys.argv[1:]))
    # Imported only now, so CLI use never loads Qt.
    from ez2digitize.app import main

    sys.exit(main())

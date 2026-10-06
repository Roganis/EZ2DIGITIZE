# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Entry point of the packaged app: the GUI, or the CLI when given a command.

EZ2DIGITIZE.AppImage                    the GUI
EZ2DIGITIZE.AppImage run ~/scans/skull  the `ez2d` CLI, headless
EZ2DIGITIZE.AppImage mask-worker ...    the masking worker (started by the app)

On Windows the same code is also `ez2d.exe`, a console program for the CLI
(EZ2DIGITIZE.exe has no console); without arguments it prints the help.
"""

import sys
from pathlib import Path

from ez2digitize import cli

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mask-worker":
        # Imported only here: ONNX Runtime loads only in the worker process.
        from ez2digitize import mask_worker

        sys.exit(mask_worker.main(sys.argv[2:]))
    if Path(sys.executable).stem.lower() == "ez2d" and len(sys.argv) == 1:
        sys.exit(cli.main(["--help"]))
    if len(sys.argv) > 1 and sys.argv[1] in (*cli.commands(), "-h", "--help"):
        sys.exit(cli.main(sys.argv[1:]))
    # Imported only now, so CLI use never loads Qt.
    from ez2digitize.app import main

    sys.exit(main())

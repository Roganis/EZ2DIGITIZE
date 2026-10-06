# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""One module per external reconstruction tool.

Each module finds its tool, checks the version against the pinned one,
builds stage command lines (as `core.stage.StageSpec`) and parses the tool's
output into progress events. Nothing here runs a stage: the pipeline does,
through `core.stage.run_stage`. Like `core`, this package never imports Qt.
"""

# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
import os

# Run Qt without a display (CI, SSH sessions). Must be set before Qt loads.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

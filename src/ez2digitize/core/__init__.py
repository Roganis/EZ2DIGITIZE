# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless application logic: projects, stages, manifests, process running.

Nothing in this package may import Qt (enforced by ruff rule TID251), so the
whole pipeline can run from a CLI and in CI without a display.
"""

# SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
# SPDX-License-Identifier: GPL-3.0-or-later
from ez2digitize.core.resources import GIB, available_memory, cpu_threads


def test_probes_return_sane_values() -> None:
    assert cpu_threads() >= 1
    assert available_memory() > 64 * 1024**2
    assert GIB == 1024**3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decide whether ``--pretty`` moved the gait, against repeats of the same command (phase2-progress §2.4).

An exact comparison cannot work: PhysX on the GPU is not bit-reproducible, so two runs of the *same* command with
the same seed already disagree slightly (measured: up to 1.0° on a mean joint angle). So the plain command is run
several times to get its own envelope, and the test is whether the ``--pretty`` run falls inside it.

Usage:
  python scripts/rsl_rl/equiv_check.py <pretty.log> <plain1.log> <plain2.log> [<plain3.log> …]
"""

from __future__ import annotations

import pathlib
import re
import sys

FLOOR = 0.05  # absolute slack in the units of the table (degrees for the angles); below this nothing is readable
SLACK = 0.5  # how far outside the plain envelope the pretty run may fall, as a fraction of the envelope's width


def numbers(path: str) -> list[float]:
    """Every number in the GAIT block: mean v_x, then the six legs x eight columns."""
    text = pathlib.Path(path).read_text()
    block = re.search(r"^GAIT .*?stance half-width\) *[-\d.]+", text, re.S | re.M)
    if block is None:
        raise SystemExit(f"no GAIT block in {path} - the run probably failed; check the log")
    return [float(v) for v in re.findall(r"-?\d+\.\d+", re.sub(r"video .*", "", block.group(0)))]


pretty = numbers(sys.argv[1])
plains = [numbers(p) for p in sys.argv[2:]]
if len(plains) < 2:
    raise SystemExit("need at least two plain runs: one alone gives no idea of the simulator's own spread")
if any(len(p) != len(pretty) for p in plains):
    raise SystemExit("the tables have different shapes; one of the runs did not finish")

lo = [min(col) for col in zip(*plains)]
hi = [max(col) for col in zip(*plains)]
width = max(top - bot for top, bot in zip(hi, lo))
over = [max(0.0, v - top, bot - v) for v, bot, top in zip(pretty, lo, hi)]
worst = max(over)
budget = max(SLACK * width, FLOOR)

print(f"plain repeats:            {len(plains)}")
print(f"their envelope, widest:   {width:.4f}   (the simulator disagreeing with itself)")
print(f"pretty outside it, worst: {worst:.4f}   in {sum(1 for o in over if o > 0)} of {len(over)} numbers")
print(f"budget = max({SLACK} x envelope, {FLOOR}) = {budget:.4f}")
if worst <= budget:
    print("\nEQUIVALENCE OK - the showcase rendering stays inside the simulator's own spread.")
else:
    print(f"\nEQUIVALENCE FAILED - {worst:.4f} > {budget:.4f}. Do not use the footage until this is explained.")
    sys.exit(1)

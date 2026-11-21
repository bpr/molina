#!/bin/bash
# Run every molina test file. Run from the repository root: pixi run test
# MOJO_FLAGS is passed to `mojo run` (e.g. "-D ASSERT=none").
set -e
# assert_almost_equal adds a default rtol=1e-5 to atol, so every call must
# pass rtol explicitly (normally rtol=0); see AGENTS.md
python3 - tests/*.mojo <<'PY'
import re, sys

bad = []
for path in sys.argv[1:]:
    src = open(path).read()
    for m in re.finditer(r"\bassert_almost_equal\(", src):
        depth, i = 1, m.end()
        while depth and i < len(src):
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        if not re.search(r"\brtol\s*=", src[m.end() : i]):
            bad.append(f"{path}:{src.count(chr(10), 0, m.start()) + 1}")
if bad:
    print("assert_almost_equal without an explicit rtol:", file=sys.stderr)
    print("\n".join("  " + b for b in bad), file=sys.stderr)
    sys.exit(1)
PY
for f in tests/test_*.mojo; do
    mojo run $MOJO_FLAGS -I . "$f"
done

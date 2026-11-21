"""Check that every public symbol of molina has a complete doc string.

`mojo doc --diagnose-missing-doc-strings` warns about public symbols without a
doc string and about doc strings without the Args, Parameters, Returns, or
Raises sections their function needs. This script fails on any warning.

    pixi run doc-check                # fail on any warning
    pixi run doc-check --list sparse  # the warnings in files matching "sparse"

Run from the repository root.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARNING = re.compile(r"^(?P<path>[^:\s]+\.mojo):(?P<line>\d+):(?P<col>\d+): warning: (?P<msg>.*)$")


def run_mojo_doc():
    """The warnings, as (relative path, line, message) tuples."""
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            ["mojo", "doc", "--diagnose-missing-doc-strings", "-o", os.path.join(tmp, "molina.json"), "molina"],
            cwd=ROOT, capture_output=True, text=True,
        )
    if proc.returncode != 0:
        sys.exit(proc.stdout + proc.stderr + "\nmojo doc failed")
    warnings = []
    for line in (proc.stdout + proc.stderr).splitlines():
        m = WARNING.match(line.strip())
        if m:
            warnings.append((os.path.relpath(m["path"], ROOT), int(m["line"]), m["msg"]))
    return sorted(warnings)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", metavar="PATTERN", default="", help="only the warnings in files whose path contains PATTERN")
    args = parser.parse_args()
    shown = [w for w in run_mojo_doc() if args.list in w[0]]
    for path, line, msg in shown:
        print(f"{path}:{line}: {msg}")
    print(f"doc string warnings: {len(shown)}")
    if shown and not args.list:
        sys.exit("doc-check failed: document the new code, see docs/writing-docs.md")


if __name__ == "__main__":
    main()

"""Candidate subprocess transport. This is process isolation, not a sandbox."""
import contextlib
import importlib.util
import json
import sys


def main():
    scenes = json.load(sys.stdin)
    spec = importlib.util.spec_from_file_location("layout_candidate", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        results = [module.solve(scene) for scene in scenes]
    sys.stdout.write(json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

"""Pytest configuration for KATE tests."""


from pathlib import Path
import shutil
import sys
import os

_PROJECT_ROOT = Path(__file__).parent
# root layout: make `import kate` work from a bare checkout without an install.
sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_sessionfinish(session, exitstatus):
    """Remove stray artifact directories left by tests using package defaults.

    Some CLI commands and runners default to writing output in cwd-relative
    directories when the path is not overridden; a test that changes cwd can leave
    these deep inside the source tree. The tree is walked only inside the project
    root, vendored and build paths are skipped, and top-level __pycache__ is cleaned
    (Python regenerates it on import, so a recursive removal is not worth the cost).
    """
    runtime_targets = {"kate_runs", "bd_sims"}
    skip_path_parts = {".git", "build", "dist", ".venv", ".github", "kate.egg-info"}

    # Safety: only walk inside the project root so that invoking pytest from
    # elsewhere (cd ~ && pytest) cannot rmtree a directory it does not own.
    project_root = _PROJECT_ROOT.resolve()
    bases = {project_root}
    cwd = Path.cwd().resolve()
    try:
        cwd.relative_to(project_root)
        bases.add(cwd)
    except ValueError:
        pass
    for base in bases:
        if not base.is_dir():
            continue
        for root, dirs, _files in os.walk(str(base), topdown=True):
            dirs[:] = [d for d in dirs if d not in skip_path_parts]
            for d in [d for d in list(dirs) if d in runtime_targets]:
                shutil.rmtree(Path(root) / d, ignore_errors=True)
                dirs.remove(d)
        pc = base / "__pycache__"
        if pc.is_dir():
            shutil.rmtree(pc, ignore_errors=True)

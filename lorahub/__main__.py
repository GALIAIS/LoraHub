"""Entry point for ``python -m lorahub``.

Mirrors what the ``lorahub`` console-script does (read from
``[project.scripts]`` in pyproject.toml). Lets ``run.{sh,bat}`` and
``lorahub manage ...`` commands invoke the CLI without depending on
PATH containing the venv's bin/ — they have an absolute path to
``python`` and can always do ``python -m lorahub ...``.
"""

from __future__ import annotations

from lorahub.cli.main import app

if __name__ == "__main__":
    app()

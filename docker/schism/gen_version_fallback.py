"""Generate schism_version.F90 for makefile builds that no longer ship gen_version.py."""

from pathlib import Path

core = Path(__file__).resolve().parent
text = (core / "schism_version.F90.template").read_text()
(core / "schism_version.F90").write_text(
    text.replace("@{VERSION_SCHISM}", "develop").replace("@{VERSION_GIT}", "none")
)

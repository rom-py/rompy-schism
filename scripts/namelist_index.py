"""Generate the namelist index for the docs (run by markdown-exec).

Every parameter rompy-schism can write to param.nml and wwminput.nml, with the
field that sets it, its default and its description, built from the namelist
models so it follows the code.
"""

from __future__ import annotations

import re
import typing

from pydantic import BaseModel

from rompy_schism.namelists import param, wwminput


def _model(annotation) -> type[BaseModel]:
    """The model in an annotation such as Optional[Proc]."""
    return next(a for a in typing.get_args(annotation) if isinstance(a, type) and issubclass(a, BaseModel))


# (file, field of NML, model, [(group, field of the model, group model)])
NAMELISTS = [
    (
        "param.nml",
        "param",
        param.Param,
        [("CORE", "core", param.Core), ("OPT", "opt", param.Opt), ("SCHOUT", "schout", param.Schout)],
    ),
    (
        "wwminput.nml",
        "wwminput",
        wwminput.Wwminput,
        [
            (name.upper(), name, _model(info.annotation))
            for name, info in wwminput.Wwminput.model_fields.items()
        ],
    ),
]

ARRAY = re.compile(r"__(\d+)$")


def _name(field: str) -> str:
    """The parameter as SCHISM reads it: iof_hydro__1 is iof_hydro(1)."""
    return ARRAY.sub(r"(\1)", field)


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _default(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return f"`'{value}'`"
    return f"`{value}`"


def namelist_index() -> str:
    """Markdown tables of the namelist parameters, one per namelist group."""
    out = []
    for filename, nml_field, model, groups in NAMELISTS:
        out.append(f"## {filename}\n")
        out.append(
            f"Set in `NML.{nml_field}`, a [`{model.__name__}`][{model.__module__}.{model.__name__}].\n"
        )
        for group, field, group_model in groups:
            fields = group_model.model_fields
            if not fields:
                continue
            path = f"{group_model.__module__}.{group_model.__name__}"
            out.append(f"### &{group}\n")
            out.append(f"`{nml_field}.{field}`, a [`{group_model.__name__}`][{path}].\n")
            out.append("| Parameter | Field | Default | Description |")
            out.append("|---|---|---|---|")
            for name, info in fields.items():
                out.append(
                    f"| `{_name(name)}` | [`{name}`][{path}.{name}] | "
                    f"{_default(info.default)} | {_cell(info.description or '')} |"
                )
            out.append("")
    return "\n".join(out)

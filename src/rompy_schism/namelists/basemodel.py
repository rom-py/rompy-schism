from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Type,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, model_serializer, model_validator

from rompy.core.types import RompyBaseModel


def get_model_field_type(
    model: Type[BaseModel], field: str
) -> Optional[Type[BaseModel]]:
    model_fields = get_type_hints(model, include_extras=True)
    annotation = model_fields.get(field)

    def find_model_type(candidate) -> Optional[Type[BaseModel]]:
        try:
            if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                return candidate
        except TypeError:
            pass

        if get_origin(candidate) is not None:
            for argument in get_args(candidate):
                model_type = find_model_type(argument)
                if model_type is not None:
                    return model_type

        return None

    return find_model_type(annotation)


def recursive_update(model: BaseModel, updates: Dict[str, Any]) -> BaseModel:
    updates_applied = model.model_dump()
    model_fields = get_type_hints(model.__class__)

    for key, value in updates.items():
        if key in model_fields:
            field_type = get_model_field_type(model.__class__, key)
            current_value = getattr(model, key)

            if isinstance(value, dict) and field_type:
                if current_value is None:
                    current_value = field_type()  # Initialize if None

                updated_value = recursive_update(current_value, value)
                updates_applied[key] = updated_value
            else:
                updates_applied[key] = value
        else:
            updates_applied[key] = value  # Extending with new fields, if any

    return model.model_copy(update=updates_applied)


class NamelistBaseModel(RompyBaseModel):
    """Base model for namelist variables"""

    namelist_name: ClassVar[Optional[str]] = None

    @model_validator(mode="before")
    def __lowercase_property_keys__(cls, values: Any) -> Any:
        def __lower__(value: Any) -> Any:
            if isinstance(value, dict):
                return {k.lower(): __lower__(v) for k, v in value.items()}
            return value

        return __lower__(values)

    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to handle proper serialization of nested components."""
        result = {}

        # Include only non-None fields in the serialized output
        for field_name in self.model_fields:
            value = getattr(self, field_name, None)
            if value is not None and not field_name.startswith("_"):
                result[field_name] = value

        return result

    def update(self, update: Dict[str, Any]):
        """Update the namelist variable with new values. Reninitializes the instance, ensuring all validations are run"""
        updated_self = recursive_update(self, update)
        updated_instance = self.__init__(**updated_self.model_dump())
        return updated_instance

    def render(self) -> str:
        """Render the namelist variable as a string"""
        return self._render_sections(self.model_dump())

    def _render_sections(self, sections: Dict[str, Any]) -> str:
        """Render an already-serialized mapping of namelist sections."""
        # create string of the form "variable = value"
        ret = []
        ret += [f"! SCHISM {self.__module__} namelist rendered from Rompy\n"]
        for section, values in sections.items():
            if values is not None:
                ret += [f"&{section}"]
                for variable, value in values.items():
                    if value is not None:
                        for ii in sorted(range(40), reverse=True):
                            variable = variable.replace(f"__{ii}", f"({ii})")
                        if isinstance(value, list):
                            value = ", ".join(
                                [self.process_value(item) for item in value]
                            )
                        else:
                            value = self.process_value(value)

                        # Check if this field needs a comma based on _comma_fields attribute
                        comma = ""
                        if (
                            hasattr(self, "_comma_fields")
                            and variable.lower() in self._comma_fields
                        ):
                            comma = ","

                        ret += [f"{variable} = {value}{comma}"]
                ret += ["/\n"]
        return "\n".join(ret)

    def process_value(self, value: Any) -> Any:
        """Process the value before rendering"""
        if isinstance(value, bool):
            value = self.boolean_to_string(value)
        elif isinstance(value, str):
            if value not in ["T", "F"]:
                value = f"'{value}'"
        return str(value)

    def boolean_to_string(self, value: bool) -> str:
        return "T" if value else "F"

    def write_nml(self, workdir: Path | str) -> None:
        """Write the namelist to a file

        Args:
            workdir (Path|str): Working directory to write to
        """
        # Ensure workdir is a Path object
        workdir_path = Path(workdir) if isinstance(workdir, str) else workdir
        filename = self.namelist_name or self.__class__.__name__.lower()
        output = workdir_path / f"{filename}.nml"
        with open(output, "w") as f:
            f.write(self.render())

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, get_type_hints

from pydantic import BaseModel

from rompy.core.types import RompyBaseModel


def get_model_field_type(
    model: Type[BaseModel], field: str
) -> Optional[Type[BaseModel]]:
    model_fields = get_type_hints(model)
    annotation = model_fields.get(field)

    if annotation:
        try:
            if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                for arg in annotation.__args__:
                    if arg is not type(None) and issubclass(arg, BaseModel):
                        return arg
            elif issubclass(annotation, BaseModel):
                return annotation
        except Exception as e:
            __import__("ipdb").set_trace()

    return None


def recursive_update(model: BaseModel, updates: Dict[str, Any]) -> BaseModel:
    updates_applied = model.dict()
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

    return model.copy(update=updates_applied)


class NamelistBaseModel(RompyBaseModel):
    """Base model for namelist variables"""

    def update(self, update: Dict[str, Any]):
        print(f"Before update: {self}")
        updated_self = recursive_update(self, update)
        updated_instance = self.__init__(**updated_self.dict())
        return updated_instance

    def render(self) -> str:
        """Render the namelist variable as a string"""
        # create string of the form "variable = value"
        ret = []
        ret += [f"! SCHISM {self.__module__} namelist rendered from Rompy\n"]
        for section, values in self.model_dump().items():
            if values is not None:
                ret += [f"&{section}"]
                for variable, value in values.items():
                    if value is not None:
                        for ii in range(13):
                            variable = variable.replace(f"__{ii}", f"({ii})")
                        if isinstance(value, list):
                            value = ", ".join([str(item) for item in value])
                        if isinstance(value, bool):
                            value = ".true." if value else ".false."
                        if isinstance(value, str):
                            value = f"{value}"
                        ret += [f"{variable} = {value}"]
                ret += ["/"]
        return "\n".join(ret)

    def write_nml(self, workdir: Path) -> None:
        """Write the namelist to a file"""
        output = workdir / f"{self.__class__.__name__.lower()}.nml"
        with open(output, "w") as f:
            f.write(self.render())

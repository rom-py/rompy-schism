import datetime
import json
import os
import re
from subprocess import run
from typing import Any

import anthropic
from pydantic import Field, root_validator, validator

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def analyze_namelist(self, content: str, filename: str) -> dict:
        prompt = f"""
        Analyze the following namelist file content from {filename} and provide a set of pydantic v2 models with the following as guidance:
        1. Each section starting with & (e.g. &example) should be a sepaate model. The name of the model should be CamelCased (eg. example becomes Example, EXAMPLE becomes Example)
        2. Scape as much as you can in terms of descriptions for each model
        3. Scrape as much as you can in terms of descriptions for each variable, noting that relevant information may not all be inline with that particular variable
        4. Construct validators for each variable where possibly using the relevant text (use the new @field_validator decorator). If distinct options are clear, use a Literal type instead of a validator
        5. Construct cross-variable validators that might be necessary (use the new @root_validator decorator)
        6. All variables should be optional (i.e using Optional[]), with None as the default
        7. Use the json_schema_extra kwarg in Field to to provide the default value for each variable as it currently is in the namelist file
        8. Finally provide a master model that takes all of the other constructed models as inputs. Names of the input variables should be lowercased version of the CamelCased model names. The name for this class should be CamelCased name of the file (whithout extension). Determine an overall summary for the docs
        9. Add doctrings to each model
        10. The master model should have a private attribute `_nml_original` that contains the full raw text of the name namelist file as provided
        11. The baseclass for all models should be rompy.schism.namelists.basemodel.NamelistBaseModel

        Please format your response as raw python code only. Don't wrap in code markers or quites or specify language
        Namelist content:
        {content}
        """

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4048,
            messages=[{"role": "user", "content": prompt}],
        )

        # Check if content is a list and join it if so
        if isinstance(message.content, list):
            content = "".join(
                [item.text for item in message.content if hasattr(item, "text")]
            )
        else:
            content = message.content
        return content


claude_client = ClaudeClient()


def generate_pydantic_models(
    filename: str,
    basemodel="rompy.schism.namelists.basemodel.NamelistBaseModel",
    claude_analysis=None,
):
    analysis = claude_analysis or {}

    with open(filename, "w") as file:
        file.write(
            f"# This file was auto generated from a schism namelist file on {datetime.datetime.now().strftime('%Y-%m-%d')}.\n\n"
        )
        file.write(f"from typing import Optional\n")
        file.write(f"from pydantic import Field, validator, root_validator\n")
        basemodellist = basemodel.split(".")
        file.write(
            f"from {'.'.join(basemodellist[0:-1])} import {basemodellist[-1]}\n\n"
        )
        file.write("\n")
        file.write(analysis)

    run(["isort", filename])
    run(["black", filename])


def nml_to_models(file_in: str, file_out: str):
    with open(file_in) as f:
        claude_analysis = claude_client.analyze_namelist(f.read(), file_in)

    generate_pydantic_models(
        file_out,
        claude_analysis=claude_analysis,
    )


def nml_to_dict(file_in: str):
    with open(file_in, "r") as file:
        input_text = file.read()
    sections = extract_sections_from_text(input_text)
    nml_dict = {}
    blurb = "\n"
    blurb += "The full contents of the namelist file are shown below providing\n"
    blurb += "associated documentation for the objects:\n\n"
    for section, text in sections.items():
        if section == "description":
            nml_dict.update({section: text})
        else:
            input_dict = extract_variables(text)
            if input_dict:
                nml_dict.update({section.lower(): input_dict})
    nml_dict["description"] = blurb + input_text
    nml_dict["full_content"] = input_text
    return nml_dict


def main():
    with open("__init__.py", "w") as f:
        for file in os.listdir("sample_inputs")[5:7]:
            components = file.split(".")
            if len(components) < 2:
                continue
            if components[1] == "nml":
                file_in = os.path.join("sample_inputs", file)
                if len(components) > 2:
                    file_out = components[0] + "_" + components[2] + ".py"
                else:
                    file_out = file.split(".")[0] + ".py"
                print(f" Processing {file_in} to {file_out}")
                nml_to_models(file_in, file_out)
                classname = file_out.split(".")[0]
                f.write(
                    f"from .{classname} import {classname.split('_')[0].capitalize()}\n"
                )
        f.write(f"from .sflux import Sflux_Inputs\n")
        f.write(f"from .schism import NML")
    run(["isort", "__init__.py"])
    run(["black", "__init__.py"])


if __name__ == "__main__":
    main()

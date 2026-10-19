"""Version contract for complete serialized SCHISM configurations."""

from typing import Literal

SchismSchemaVersion = Literal["schism-v5.13", "schism-v5.14"]

# This default is a permanent compatibility contract. Never advance it when a
# new SCHISM schema is introduced; unversioned historical configs are v5.13.
DEFAULT_SCHISM_SCHEMA: SchismSchemaVersion = "schism-v5.13"

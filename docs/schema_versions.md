# SCHISM configuration schema versions

`SCHISMConfig.schema_version` identifies the contract used by the **complete
serialized configuration**. It is the single source of truth for
version-dependent defaults, validation, serialization, and generated model
inputs.

```yaml
model_type: schism
schema_version: schism-v5.14

grid:
  # ...

nml:
  param:
    # No nested version discriminator
```

## Compatibility contract

Configurations that omit `schema_version` are always interpreted as
`schism-v5.13`. This is a permanent compatibility rule: the implicit version
must not advance when support for newer SCHISM releases is added.

When an unversioned configuration is serialized, Rompy writes the resolved
value explicitly:

```yaml
schema_version: schism-v5.13
```

This ensures that loading and re-serializing a historical configuration makes
its contract visible without changing its behaviour.

## Supported schemas

| Schema | `CORE.nmarsh_types` | `OPT.isconsv` |
| --- | --- | --- |
| `schism-v5.13` | Not accepted or emitted | Accepted and emitted |
| `schism-v5.14` | Defaults to `2` and is emitted | Not emitted; non-zero values are rejected |

For v5.14, precipitation and evaporation are selected when SCHISM is compiled
with `PREC_EVAP`; they are not selected through `isconsv`.

Serialized configurations contain the resolved fields for their selected
schema. Consequently, a load → dump → load round trip preserves both the schema
version and the generated namelist behaviour.

## Adding a future schema

Support for a new SCHISM schema must:

1. add an explicit `schema_version` value;
2. leave the unversioned default fixed at `schism-v5.13`;
3. define all changed defaults, validation, serialization, and rendering rules;
4. provide an explicit migration for any changed meaning;
5. add complete YAML round-trip tests; and
6. test generated inputs against a matching SCHISM revision.

Do not independently version nested namelist models. A SCHISM release is one
configuration contract, even when a particular release changes only one
namelist.

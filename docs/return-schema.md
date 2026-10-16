# Return-schema reference

`SCHISMConfig.expected_artifacts()` returns core `Artifact` objects with
workspace-relative POSIX paths. It does not read input datasets or run SCHISM.
`validate_outputs(workspace)` returns existing files with byte sizes and warns
for missing declarations. Missing files are **not** observations, allowing core
`ModelRun.run()` to populate `expected_outputs` and `missing_outputs` correctly.
Discovery also includes input/control files, diagnostics and auxiliary outputs;
it excludes core result sidecars, `.rompy-postprocess` and symlinks escaping the
workspace.

## Declared outputs

- `outputs/mirror.out`.
- Default scribed I/O: `outputs/out2d_<stack>.nc`,
  `outputs/zCoordinates_<stack>.nc` (solver default `iof_hydro(25)=1`), and
  separate files for enabled 3D hydrodynamic variables. Vector components have
  separate X/Y files. `nc_out=0` disables these declarations.
- `outputs/hotstart_000000_<step>.nc` when `nhot=1`, at each `nhot_write`
  interval. Only rank zero is predictable without backend MPI metadata.

Stack predictions use `rnday`, `dt`, `nspool` and `ihfskip` from the namelist.
A partial last stack is declared only if it contains an output timestep.
`ipre=1` and continued-clock restarts (`ihot=2`) declare only the log; the latter
requires a clock from `hotstart.nc` that is not available in config alone.

## Discovery-only outputs and limitations

Station switches reside in `station.in`, so `staout_*` files are discovered, not
promised as a fixed set. Other MPI hotstart ranks, combined hotstarts, optional
module outputs and OLDIO `schout_*` files are also discovered. The default
expectations assume a scribed-I/O executable, matching this repository's Docker
builds; OLDIO builds need a config override for a complete expected manifest.
The legacy `SchismCSIROConfig` declares only `outputs/mirror.out` and discovers
other files because it does not expose the modern complete namelist contract.

File presence is not a scientific/content validation or a completion check.
SCHISM can exit zero after a model error; consult `outputs/mirror.out` and
`outputs/fatal.error` as described in `docker/README.md`.

Output conventions: [SCHISM output reference](https://schism-dev.github.io/schism/master/input-output/outputs.html).

# Patches to the SCHISM source

Applied in order to the SCHISM release before it is built. Each fixes a problem
that is still in the SCHISM version the image builds; remove it when a release
includes the fix.

- `wwm-boundary-spectra-on-node.patch`: WWM interpolates WAVEWATCH III boundary
  spectra (`IBOUNDFORMAT=6`) to each wave boundary node from the two nearest
  spectra, weighted by `1/distance`. A node exactly on a spectra point gets an
  infinite weight and no waves. With the patch, such a node takes that point's
  spectrum. This is common: meshes often have boundary nodes on round
  coordinates, as do spectra from a regular grid.

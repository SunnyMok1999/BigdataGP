# MaleCNS v1.0 circuit subset

Data creators: FlyEM / HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology and Google Research. Dataset/project: https://male-cns.janelia.org/download/

License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/. No endorsement of this experiment is implied.

Derived from the v1.0 minimum-confidence-0.5 annotation and connectivity tables, and v1.0 neurotransmitter table. Exact URLs and SHA-256 are documented in the repository protocol and manifest.

Changes: deterministic selection of 80 cells and all 1,296 measured internal directed edges; retention of original body IDs, soma coordinates, type and transmitter annotations; creation of graph indices and engineered input/output roles. Source edge contact counts remain unchanged. The runtime normalizes incoming contact counts and applies assumed transmitter signs and leaky tanh dynamics. These transformations are not biological measurements.

The graph represents a small selected circuit, not a complete brain. Its activity is simulated and dimensionless. Input encoding and action readout are artificial.

Reproduce: `uv run --with pyarrow --with numpy python scripts/build-connectome.py /tmp/pinfly-data`. Source hashes are checked before extraction. See docs/experiment.md for download commands, exact equations, boundary assumptions and validation.

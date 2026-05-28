SAM2 Runtime Fix
================

This build changes SAM2 installation so hydra-core, omegaconf and antlr4 are installed into:

  runtime/sam2/pydeps

The runner prepends this folder to sys.path before importing SAM2. This prevents QGIS Python from using an old/incompatible hydra-core from the global QGIS environment.

Why this is needed:
- QGIS/Python 3.12 can fail when pip tries to unpack antlr4-python3-runtime 4.9.3 from source.
- A previous fallback installed hydra-core 0.11.3, which is too old for SAM2 and misses initialize_config_module.
- This build installs hydra-core 1.3.2 and omegaconf 2.3.0 as isolated runtime wheels with --no-deps.

Usage:
1. Install the plugin ZIP in QGIS.
2. Open Annotator > SAM2.
3. Press "Install SAM2 automatically" once.
4. Then use "SAM2 current image" or "SAM2 all pictures".

Default model is base_plus: sam2_hiera_base_plus.pt with sam2_hiera_b+.yaml.

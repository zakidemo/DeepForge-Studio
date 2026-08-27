# Test suite

Regression tests for the code generator. They exist because v1.0.3 shipped
model types that the interface offered but the generator could not produce.

## Run

    node tests/harness.mjs        # generate every UI-reachable configuration
    python3 tests/run_all.py     # run every export end to end against real data
    node tests/ui_config_test.mjs # layer modal, skip control, API-key leakage
    node tests/capsule_test.mjs   # capsule is a valid, readable ZIP
    node tests/model_selection_test.mjs # provider model discovery and fallback
    node tests/sweep.mjs          # generate every ML parameter combination
    python3 tests/sweep_exec.py   # syntax-check all, execute a sample
    

Requires `tensorflow-cpu`, `scikit-learn`, `xgboost`, `pillow`.

## What they assert

- Every model in `js/config/models.js` has a generator (`mlBuilders` coverage).
- Every layer in `js/config/layers.js` emits code (no silent drops).
- Every generated script is valid Python.
- Every model constructs under TensorFlow/Keras or scikit-learn.
- No ML parameter combination produces a script that fails at run time.
- Configurations that cannot produce a valid model refuse to export.
- Every layer type opens a configuration modal.
- Skip connections offer only genuinely earlier layers as sources.
- No exported artefact contains the API key.
- The capsule is a valid ZIP with all six expected members.
- No provider model name is hardcoded; discovery falls back sensibly.

## Known exclusions

- The six `pretrained` exports download ImageNet weights, so they are reported
  as BLOCKED where `storage.googleapis.com` is unreachable. Run them in an
  environment with network access before reporting results.

## Current status (v2.0.0)

    29 pass / 0 fail / 6 blocked / 35 total
    10/10 classical ML models generate and run
    10/10 Custom Builder layer types emit code
    974/974 ML parameter combinations valid

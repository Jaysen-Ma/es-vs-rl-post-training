# Optional shell settings for the external maths evaluation harness.
# Copy to config.sh (ignored by Git); scripts/passk/run_eval.sh sources it.
# Python runners do not source this file: export values in your shell or source
# config.sh yourself. Model paths for all workflows belong in config.toml.

export MATH_EVAL_DIR=/path/to/limit-of-RLVR/math/examples/math_eval
export ES_CAPACITY_CONFIG=/absolute/path/to/config.toml

# Unpacked ETHICS dataset. run_rest.py also accepts --data-dir.
# export ETHICS_DATA_DIR=/path/to/ethics/ethics

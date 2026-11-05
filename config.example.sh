# Copy this file to config.sh (gitignored) and fill in the paths for your
# environment. scripts/passk/run_eval.sh sources config.sh automatically if it
# exists. Nothing here is assumed by default — on a box that already
# followed docker/on_start.sh's layout, run_eval.sh falls back to
# $WORKSPACE/limit-of-RLVR/math/examples/math_eval if config.sh is absent
# and MATH_EVAL_DIR isn't otherwise set; everywhere else, set it explicitly.

# Where Jaysen-Ma/limit-of-RLVR (fix/math-equal-timeout-bypass) is checked
# out — scripts/passk/run_eval.sh drives its math_eval harness directly.
export MATH_EVAL_DIR=/path/to/limit-of-RLVR/math/examples/math_eval

# One variable per model — no shared "models root" is assumed, since a
# checkpoint you trained yourself often doesn't live next to the base models
# you downloaded. Pass the variable name as run_eval.sh's first argument,
# e.g. `run_eval.sh QWEN25_7B_BASE base my-run`, and it resolves to the path
# below; an absolute path passed directly always works too, config or not.
# Names are yours to choose — these match this project's own base/ES/RL
# arms at both scales as a starting point.
export QWEN25_1_5B_BASE=/path/to/models/Qwen2.5-1.5B
export QWEN25_1_5B_ES=/path/to/models/Qwen2.5-1.5B-ES-math
export QWEN25_1_5B_RL=/path/to/models/Qwen-2.5-1.5B-SimpleRL-Zoo
export QWEN25_7B_BASE=/path/to/models/Qwen2.5-7B
export QWEN25_7B_ES=/path/to/models/Qwen2.5-7B-ES-math
export QWEN25_7B_RL=/path/to/models/Qwen-2.5-7B-SimpleRL-Zoo

# Python evaluation runners use MODELS_DIR for local mirrors (default ~/models).
# They do not source this file automatically: export these values in your shell
# or source config.sh before running a Python command.
# export MODELS_DIR=/path/to/models

# Unpacked ETHICS subsets; run_rest.py also accepts --data-dir, which takes
# precedence over this variable. The default remains /tmp/ethics/ethics.
# export ETHICS_DATA_DIR=/path/to/ethics/ethics

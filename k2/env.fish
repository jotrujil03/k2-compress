# k2/env.fish
# Source this when working on K2:
#   source ~/Documents/k2-compress/k2/env.fish

set -x PYTHONPATH /home/jotrujil/Documents/k2-compress/.venv/lib/python3.14/site-packages
set -x K2_PYTHON_DIR /home/jotrujil/Documents/k2-compress/k2/src/python

echo "K2 env loaded"
echo "  PYTHONPATH = $PYTHONPATH"
echo "  K2_PYTHON_DIR = $K2_PYTHON_DIR"

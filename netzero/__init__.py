"""Net-zero extension: device-agnostic, carbon-aware control on top of the fuzzy risk engine.

Modules
-------
fuzzy       crop-specific fuzzy evaluation (wraps fuzzyLozic)
facility    function-based device profiles, candidate generation, command building
model       learned environment dynamics (recursive least squares)
grid        hourly grid carbon intensity and electricity price
simulator   virtual smart farm with hidden physical parameters
controllers threshold baseline and fuzzy optimizers (legacy / learned / carbon-aware)
experiment  scenario runner, metrics, CLI
"""

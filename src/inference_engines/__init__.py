"""Inference engines, explained by running them.

The package is split by question:

* ``config`` / ``client``  - how do I talk to an engine? (all four speak the OpenAI API)
* ``engines``              - how do I launch each one, and what is engine specific?
* ``concepts``             - what does each engine actually do inside? (pure-Python simulations)
* ``benchmark``            - how do I measure it under *my* workload?
* ``decision``             - which one should I choose, and when?
"""

__version__ = "0.1.0"

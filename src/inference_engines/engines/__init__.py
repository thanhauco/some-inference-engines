"""Engine-specific knowledge: how to launch each engine and what it offers beyond the OpenAI API.

Every module exposes a ``launch_command(...) -> list[str]`` that builds the exact command
line for that engine, so the flags that matter are documented in code and tested.
"""

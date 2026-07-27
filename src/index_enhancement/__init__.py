"""Point-in-time index-universe enhancement utilities.

Imports stay lazy so ``python -m src.index_enhancement.universe`` can run
without pre-importing the target module or emitting a runpy warning.
"""

from importlib import import_module

__all__ = [
    "INDEX_SPECS",
    "build_all_index_inputs",
    "build_index_input",
    "fetch_and_save_components",
    "fetch_and_save_weights",
    "load_index_weights",
    "load_components",
    "build_forward_labels",
    "build_index_labels",
    "build_all_index_labels",
]


def __getattr__(name: str):
    if name == "INDEX_SPECS" or name in {"fetch_and_save_components", "load_components"}:
        return getattr(import_module(".universe", __name__), name)
    if name in {"build_all_index_inputs", "build_index_input"}:
        return getattr(import_module(".builder", __name__), name)
    if name in {"fetch_and_save_weights", "load_index_weights"}:
        return getattr(import_module(".weights", __name__), name)
    if name in {"build_forward_labels", "build_index_labels", "build_all_index_labels"}:
        return getattr(import_module(".labels", __name__), name)
    raise AttributeError(name)

"""Method registry and constructor."""

from pathlib import Path
from typing import Any

from .core import AuxiliaryCall, MemoryMethod
from .rag import RAGMethod
from .reflexion import ReflexionMethod
from .mem0 import Mem0Method
from .memp import MemPMethod
from .memrl import MemRLMethod
from .memskill import MemSkillMethod
from .evosc import EvoSCMethod
from .evoskill import EvoSkillMethod
from .tokenworth import TokenWorthMethod
from .shared_retrieval import HASH_COSINE_METHODS, HASH_COSINE_VARIANT


class NoMemoryMethod(MemoryMethod):
    method_name = "NoMemory"


METHODS: dict[str, type[MemoryMethod]] = {
    "NoMemory": NoMemoryMethod,
    "RAG": RAGMethod,
    "Reflexion": ReflexionMethod,
    "Mem0": Mem0Method,
    "MemP": MemPMethod,
    "MemRL": MemRLMethod,
    "MemSkill": MemSkillMethod,
    "EvoSC": EvoSCMethod,
    "EvoSkill": EvoSkillMethod,
    "TokenWorth": TokenWorthMethod,
}


def create_method(
    name: str,
    state_dir: Path,
    config: dict[str, Any],
    auxiliary_call: AuxiliaryCall | None = None,
) -> MemoryMethod:
    try:
        method_class = METHODS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown memory method {name!r}; choose from {sorted(METHODS)}"
        ) from error
    if config.get("retrieval_variant") == HASH_COSINE_VARIANT:
        method_class = HASH_COSINE_METHODS.get(name, method_class)
    return method_class(state_dir, config, auxiliary_call)

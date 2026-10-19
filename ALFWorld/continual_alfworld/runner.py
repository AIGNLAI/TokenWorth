"""Unified continual ALFWorld runner with configurable memory-update phases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from tokenworth_core.runtime import (
    CheckpointStore,
    RunGuard,
    atomic_json,
    fingerprint,
    redacted_config,
    source_identity,
)
from .client import OpenAICompatibleClient
from .loop_guard import detect_loop, loop_break_navigation_candidates
from .methods import METHODS, EpisodeOutcome, RetrievedMemory, create_method
from .retrieval import truncate


METHOD_REFERENCE_SYSTEM_PROMPT = """You are an embodied agent in the ALFWorld
text environment. Complete the CURRENT household task efficiently by grounding
every decision in the CURRENT episode.

A completed past episode may be provided before the current task. Its only
permitted use is as a METHOD reference: learn a general process for turning an
observation into a correct next step. It is never evidence about the current
world. Past goals, target objects, object numbers, locations, layouts,
inventory, receptacle contents, and open/closed or processed states must not be
carried into the current episode. Entity numbers are local to each episode.

At every step, first identify the exact target and requirement from the CURRENT
task, then use the CURRENT observation and CURRENT in-episode history to infer
what is known, and finally choose exactly one command from the CURRENT
admissible-actions list. Never use a substitute or placeholder object. Never
assume a past location; if the target has not been observed, search unvisited
locations using current evidence. If past experience conflicts with or is not
supported by current evidence, ignore it completely.

Common prerequisites: locate and take the requested object before manipulating
it; open closed containers; after cleaning, heating, or cooling, place that same
object in the requested destination. For examination under a lamp, hold the
requested object, go to a lamp, switch it on, and examine the held object.

Return one JSON object and nothing else:
{"reasoning":"brief reason","action":"exact admissible command"}
"""

TASK_PREFIXES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

SPLIT_NAMES = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def memory_update_phases(config: Dict[str, Any]) -> Tuple[str, ...]:
    """Return phases that update method state, preserving frozen eval by default."""

    configured = config.get("memory", {}).get("update_phases", ["train"])
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError("memory.update_phases must be a non-empty list")
    phases = tuple(str(item) for item in configured)
    unknown = sorted(set(phases) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError("unknown memory.update_phases: " + ", ".join(unknown))
    if len(set(phases)) != len(phases):
        raise ValueError("memory.update_phases must not contain duplicates")
    return phases


def is_memory_update_phase(config: Dict[str, Any], phase: str) -> bool:
    return phase in memory_update_phases(config)


def normalize_action(text: str) -> str:
    return " ".join(text.strip().lower().rstrip(".").split())


def extract_action(response: str, actions: Sequence[str]) -> Tuple[str, str, bool]:
    proposed = ""
    reasoning = ""
    start, end = response.find("{"), response.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(response[start : end + 1])
            proposed = str(parsed.get("action", ""))
            reasoning = str(parsed.get("reasoning", ""))
        except json.JSONDecodeError:
            pass
    if not proposed:
        match = re.search(
            r"(?:action\s*:|<action>)(.*?)(?:</action>|$)",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            proposed = match.group(1).strip().strip("`\"'{}[] ")
    normalized = {normalize_action(action): action for action in actions}
    if normalize_action(proposed) in normalized:
        return normalized[normalize_action(proposed)], reasoning, True
    response_normalized = normalize_action(response)
    embedded = [
        action for action in actions if normalize_action(action) in response_normalized
    ]
    if embedded:
        return max(embedded, key=len), reasoning, True
    fallback = "look" if "look" in actions else actions[0]
    return fallback, reasoning, False


def task_type_from_gamefile(gamefile: str) -> str:
    task_directory = Path(gamefile).parent.parent.name
    for prefix in TASK_PREFIXES:
        if task_directory.startswith(prefix + "-"):
            return prefix
    return "unknown"


def task_from_observation(observation: str) -> str:
    marker = "Your task is to:"
    if marker in observation:
        return observation.split(marker, 1)[1].strip()
    return observation.strip()


def select_games(
    game_files: Sequence[str], count: int, strategy: str, seed: int
) -> List[str]:
    files = sorted(str(item) for item in game_files)
    if count == 0 or count < -1:
        raise ValueError("game count must be -1 (all) or a positive integer")
    requested = len(files) if count == -1 else count
    if requested > len(files):
        raise ValueError(
            "requested {} games, split contains {}".format(requested, len(files))
        )
    rng = random.Random(seed)
    if strategy == "first":
        return files[:requested]
    if strategy == "random":
        rng.shuffle(files)
        return files[:requested]
    if strategy != "stratified":
        raise ValueError("unknown sampling strategy: " + strategy)
    groups: Dict[str, List[str]] = {prefix: [] for prefix in TASK_PREFIXES}
    other: List[str] = []
    for gamefile in files:
        task_type = task_type_from_gamefile(gamefile)
        if task_type in groups:
            groups[task_type].append(gamefile)
        else:
            other.append(gamefile)
    for group in groups.values():
        rng.shuffle(group)
    rng.shuffle(other)
    selected: List[str] = []
    while len(selected) < requested:
        made_progress = False
        for prefix in TASK_PREFIXES:
            if groups[prefix] and len(selected) < requested:
                selected.append(groups[prefix].pop())
                made_progress = True
        if other and len(selected) < requested:
            selected.append(other.pop())
            made_progress = True
        if not made_progress:
            break
    if len(selected) != requested:
        raise RuntimeError("could not construct stratified sample")
    return selected


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge a small experiment variant over its base config."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    extends = config.pop("extends", None)
    if extends:
        base_path = (path.parent / str(extends)).resolve()
        if base_path == path.resolve():
            raise ValueError("config cannot extend itself: {}".format(path))
        config = _merge_config(load_config(base_path), config)
    return config


def model_environment(model: Dict[str, Any]) -> Dict[str, str]:
    """Read an optional .env file without exporting or persisting its secrets."""

    values: Dict[str, str] = {}
    env_file = str(model.get("env_file", "")).strip()
    if env_file:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise ValueError("model.env_file does not exist: " + env_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            if item.startswith("export "):
                item = item[7:].strip()
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key:
                values[key] = value
    values.update(os.environ)
    return values


def resolve_model_environment(config: Dict[str, Any]) -> None:
    """Resolve cloud endpoint/model without copying API secrets into config."""

    model = config["model"]
    values = model_environment(model)
    for field, env_field in (("base_url", "base_url_env"), ("name", "name_env")):
        env_name = str(model.get(env_field, "")).strip()
        if env_name:
            value = values.get(env_name, "").strip()
            if not value:
                raise ValueError(
                    "model.{} requires environment variable {}".format(
                        env_field, env_name
                    )
                )
            model[field] = value
    api_key_env = str(model.get("api_key_env", "")).strip()
    if api_key_env and not values.get(api_key_env, "").strip():
        raise ValueError(
            "model.api_key_env requires environment variable " + api_key_env
        )


def resolve_environment_paths(config: Dict[str, Any]) -> None:
    """Resolve portable ALFWorld paths without persisting environment values."""

    environment = config["environment"]
    values = model_environment(config.get("model", {}))
    for field, env_field in (
        ("alfworld_config", "alfworld_config_env"),
        ("data_root", "data_root_env"),
    ):
        env_name = str(environment.get(env_field, "")).strip()
        if not env_name:
            continue
        value = values.get(env_name, "").strip()
        if not value:
            raise ValueError(
                "environment.{} requires environment variable {}".format(
                    env_field, env_name
                )
            )
        environment[field] = value


def resolve_api_key(model_config: Dict[str, Any]) -> str:
    env_name = str(model_config.get("api_key_env", "")).strip()
    if env_name:
        value = model_environment(model_config).get(env_name, "").strip()
        if not value:
            raise ValueError(
                "model.api_key_env requires environment variable " + env_name
            )
        return value
    return str(model_config.get("api_key", "local"))


def load_alfworld_config(config: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(str(config["environment"]["alfworld_config"])).expanduser()
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    data_root = Path(str(config["environment"]["data_root"])).expanduser().resolve()
    result["dataset"]["data_path"] = str(data_root / "json_2.1.1" / "train")
    result["dataset"]["eval_id_data_path"] = str(
        data_root / "json_2.1.1" / "valid_seen"
    )
    result["dataset"]["eval_ood_data_path"] = str(
        data_root / "json_2.1.1" / "valid_unseen"
    )
    result["dataset"]["num_train_games"] = -1
    result["dataset"]["num_eval_games"] = -1
    result["logic"]["domain"] = str(data_root / "logic" / "alfred.pddl")
    result["logic"]["grammar"] = str(data_root / "logic" / "alfred.twl2")
    result["general"][
        "training_method"
    ] = "dqn"  # prevents expert-plan leakage in train
    result["general"]["random_seed"] = int(config["experiment"]["seed"])
    result["env"]["domain_randomization"] = False
    result["rl"]["training"]["max_nb_steps_per_episode"] = int(
        config["agent"]["max_steps"]
    )
    return result


def prepare_manifest(config: Dict[str, Any], run_root: Path) -> Dict[str, List[str]]:
    manifest_path = run_root / "game_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {key: list(value) for key, value in payload["games"].items()}
    from alfworld.agents.environment import get_environment

    base_config = load_alfworld_config(config)
    games: Dict[str, List[str]] = {}
    seed = int(config["experiment"]["seed"])
    for offset, phase in enumerate(("train", "valid_seen", "valid_unseen")):
        phase_config = config["phases"][phase]
        if not bool(phase_config.get("enabled", True)):
            games[phase] = []
            continue
        wrapper = get_environment("AlfredTWEnv")(
            copy.deepcopy(base_config), train_eval=SPLIT_NAMES[phase]
        )
        games[phase] = select_games(
            wrapper.game_files,
            int(phase_config["games"]),
            str(phase_config.get("sampling", "stratified")),
            seed + offset,
        )
    payload = {
        "seed": seed,
        "games": games,
        "counts": {phase: len(values) for phase, values in games.items()},
        "sha256": hashlib.sha256(
            json.dumps(games, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return games


def format_turn(
    task: str,
    observation: str,
    actions: Sequence[str],
    history: Sequence[Dict[str, str]],
    memory_context: str,
    history_turns: int,
    observation_chars: int,
    prompt_format: str = "method_reference",
) -> str:
    history_lines = []
    for index, turn in enumerate(history[-history_turns:], 1):
        history_lines.append(
            "{}. Action: {}\n   Result: {}".format(
                index,
                turn["action"],
                truncate(turn["observation"], observation_chars),
            )
        )
    if prompt_format != "method_reference":
        raise ValueError("unknown agent.memory_prompt_format: " + prompt_format)
    return """PAST EPISODE — METHOD REFERENCE ONLY
The following completed history may suggest a general observation-to-action
method. Do not reuse any of its task goals, objects, numbers, locations,
inventory, layout, or environment state.
{memory}

END OF PAST EPISODE
Reminder: Everything above is method reference only. The concrete task and
environment below are a different CURRENT episode and are authoritative. Base
the next action on the current task, current observations, current in-episode
history, and current admissible actions. Do not copy a past action or use a
placeholder object.

CURRENT EPISODE — AUTHORITATIVE
Current task:
{task}

Previous turns from this CURRENT episode:
{history}

Current observation:
{observation}

Current admissible actions:
{actions}

Before answering, verify that the action advances the exact CURRENT target and
is supported by CURRENT evidence. Choose one exact admissible action and return
valid JSON only.""".format(
        task=task,
        memory=memory_context,
        history="\n".join(history_lines) if history_lines else "(none)",
        observation=truncate(observation, observation_chars),
        actions="\n".join("- " + action for action in actions),
    )


def choose_action(
    client: OpenAICompatibleClient,
    config: Dict[str, Any],
    task: str,
    observation: str,
    actions: Sequence[str],
    history: Sequence[Dict[str, str]],
    memory_context: str,
) -> Dict[str, Any]:
    model_config = config["model"]
    agent_config = config["agent"]
    prompt = format_turn(
        task,
        observation,
        actions,
        history,
        memory_context,
        int(agent_config.get("history_turns", 12)),
        int(agent_config.get("observation_chars", 1800)),
        str(agent_config.get("memory_prompt_format", "method_reference")),
    )
    system_prompt = METHOD_REFERENCE_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    attempts = 1 + int(agent_config.get("repair_attempts", 1))
    responses = []
    action = "look" if "look" in actions else actions[0]
    reasoning = ""
    valid = False
    for attempt in range(attempts):
        purpose = "action" if attempt == 0 else "action_repair"
        response = client.complete(
            messages,
            purpose=purpose,
            temperature=float(model_config.get("temperature", 0.0)),
            top_p=float(model_config.get("top_p", 1.0)),
            max_tokens=int(model_config.get("action_max_tokens", 128)),
        )
        responses.append(response)
        action, reasoning, valid = extract_action(response, actions)
        if valid:
            break
        messages.extend(
            [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": "The proposed action is not admissible. Choose exactly one command below and return JSON only.\n"
                    + "\n".join("- " + candidate for candidate in actions),
                },
            ]
        )
    store_raw = bool(config.get("logging", {}).get("store_raw_responses", False))
    return {
        "action": action,
        "reasoning": reasoning,
        "valid": valid,
        "repair_used": len(responses) > 1,
        "raw_responses": responses if store_raw else None,
    }


def subtract_stats(
    after: Dict[str, Dict[str, float]], before: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for purpose in set(after) | set(before):
        keys = set(after.get(purpose, {})) | set(before.get(purpose, {}))
        values = {
            key: float(after.get(purpose, {}).get(key, 0.0))
            - float(before.get(purpose, {}).get(key, 0.0))
            for key in keys
        }
        for integer_key in ("calls", "prompt_tokens", "completion_tokens", "errors"):
            if integer_key in values:
                values[integer_key] = int(values[integer_key])
        result[purpose] = values
    return result


def run_episode(
    env: Any,
    client: OpenAICompatibleClient,
    method: Any,
    config: Dict[str, Any],
    phase: str,
    epoch: int,
    episode_index: int,
    expected_gamefile: Optional[str] = None,
) -> Tuple[Dict[str, Any], EpisodeOutcome]:
    observations, infos = env.reset()
    actual_gamefile = str(infos["extra.gamefile"][0])
    if (
        expected_gamefile is not None
        and Path(actual_gamefile).resolve() != Path(expected_gamefile).resolve()
    ):
        raise RuntimeError("Environment game order differs from frozen manifest")
    observation = str(observations[0])
    task = task_from_observation(observation)
    gamefile = str(infos["extra.gamefile"][0])
    task_type = task_type_from_gamefile(gamefile)
    method.learning = is_memory_update_phase(config, phase)
    method.set_retrieval_context(observation)
    retrieved: List[RetrievedMemory] = method.retrieve(task, task_type)
    memory_context = method.render_context(retrieved)
    initial_memory_context_chars = len(memory_context) if retrieved else 0
    memory_disabled_after_loop = False
    loop_guard_reason: Optional[str] = None
    loop_guard_step: Optional[int] = None
    true_loop_break_enabled = bool(
        config["agent"].get("true_loop_break_enabled", False)
    )
    force_loop_break_next = False
    loop_break_trigger_count = 0
    loop_break_forced_actions = 0
    visited_navigation_counts: Dict[str, int] = {}
    history: List[Dict[str, str]] = []
    turns: List[Dict[str, Any]] = []
    won = False
    before_stats = client.snapshot_stats()
    started = time.perf_counter()
    max_steps = int(config["agent"]["max_steps"])
    for step_index in range(max_steps):
        actions = [str(item) for item in infos["admissible_commands"][0]]
        decision_actions = actions
        loop_break_forced = False
        if force_loop_break_next:
            candidates = loop_break_navigation_candidates(
                actions, visited_navigation_counts
            )
            if candidates:
                decision_actions = list(candidates)
                loop_break_forced = True
                loop_break_forced_actions += 1
            force_loop_break_next = False
        decision = choose_action(
            client,
            config,
            task,
            observation,
            decision_actions,
            history,
            memory_context,
        )
        observations, _, dones, infos = env.step([decision["action"]])
        next_observation = str(observations[0])
        external_won = infos["won"][0]
        if not isinstance(external_won, (bool, int)) or external_won not in (
            False,
            True,
        ):
            raise RuntimeError("Environment returned an invalid success flag")
        won = bool(external_won)
        done = bool(dones[0])
        turns.append(
            {
                "step": step_index + 1,
                "observation": observation,
                "action": decision["action"],
                "reasoning": decision["reasoning"],
                "valid": decision["valid"],
                "repair_used": decision["repair_used"],
                "raw_responses": decision["raw_responses"],
                "loop_break_forced": loop_break_forced,
                "next_observation": next_observation,
                "won": won,
                "done": done,
            }
        )
        history.append({"action": decision["action"], "observation": next_observation})
        normalized_decision = normalize_action(decision["action"])
        if normalized_decision.startswith("go to "):
            visited_navigation_counts[normalized_decision] = (
                visited_navigation_counts.get(normalized_decision, 0) + 1
            )
        if bool(config["agent"].get("loop_guard_enabled", False)) and (
            (retrieved and not memory_disabled_after_loop) or true_loop_break_enabled
        ):
            loop_detection = detect_loop(
                history,
                window=int(config["agent"].get("loop_guard_window", 6)),
                repeat_limit=int(config["agent"].get("loop_guard_repeat_limit", 3)),
            )
            if loop_detection:
                if loop_guard_step is None:
                    loop_guard_reason = loop_detection.reason
                    loop_guard_step = step_index + 1
                if retrieved and not memory_disabled_after_loop:
                    memory_disabled_after_loop = True
                    if bool(config["agent"].get("disable_memory_after_loop", True)):
                        memory_context = (
                            "(The retrieved past memory has been disabled because it led to a "
                            "repeated-action loop. Ignore its route and explore an unvisited "
                            "location relevant to the current target.)"
                        )
                if true_loop_break_enabled:
                    loop_break_trigger_count += 1
                    force_loop_break_next = True
        observation = next_observation
        if done:
            break
    episode_id = "{}-e{}-{:05d}".format(phase, epoch, episode_index)
    outcome = EpisodeOutcome(
        phase=phase,
        epoch=epoch,
        episode_id=episode_id,
        task=task,
        task_type=task_type,
        gamefile=gamefile,
        success=won,
        steps=len(turns),
        transcript=[
            {
                "observation": item["observation"],
                "action": item["action"],
                "next_observation": item["next_observation"],
            }
            for item in turns
        ],
        memory_disabled_after_loop=memory_disabled_after_loop,
        loop_break_trigger_count=loop_break_trigger_count,
        loop_break_forced_actions=loop_break_forced_actions,
    )
    result = {
        "episode_id": episode_id,
        "phase": phase,
        "epoch": epoch,
        "episode_index": episode_index,
        "task": task,
        "task_type": task_type,
        "gamefile": gamefile,
        "success": won,
        "steps": len(turns),
        "invalid_actions": sum(not item["valid"] for item in turns),
        "repair_calls": sum(item["repair_used"] for item in turns),
        "wall_seconds": time.perf_counter() - started,
        "retrieved": [item.__dict__ for item in retrieved],
        "retrieved_context_chars": initial_memory_context_chars,
        "gate_info": copy.deepcopy(getattr(method, "last_gate_info", None)),
        "memory_disabled_after_loop": memory_disabled_after_loop,
        "loop_guard_reason": loop_guard_reason,
        "loop_guard_step": loop_guard_step,
        "true_loop_break_enabled": true_loop_break_enabled,
        "loop_break_trigger_count": loop_break_trigger_count,
        "loop_break_forced_actions": loop_break_forced_actions,
        "llm": None,
        "turns": turns,
    }
    # Learning happens after action generation. Its auxiliary calls belong to
    # this episode's accounting but can never influence the completed episode.
    if method.learning:
        method.observe(outcome)
    elif hasattr(method, "finish_without_update"):
        method.finish_without_update()
    result["llm"] = subtract_stats(client.snapshot_stats(), before_stats)
    result["wall_seconds"] = time.perf_counter() - started
    return result, outcome


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed(path: Path) -> Dict[str, Dict[str, Any]]:
    result = {}
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "invalid JSONL at {}:{}: {}".format(path, line_number, error)
                )
            key = str(item["episode_id"])
            if key in result:
                raise RuntimeError("Duplicate completed episode: " + key)
            result[key] = item
    return result


def reconcile_pending(method: Any, trajectory_path: Path) -> None:
    pending = method.state.get("_runner_pending_result")
    if not pending:
        return
    completed = load_completed(trajectory_path)
    episode_id = str(pending["episode_id"])
    if episode_id not in completed:
        append_jsonl(trajectory_path, pending)
    method.state.pop("_runner_pending_result", None)
    method.save()


def create_env(
    config: Dict[str, Any], phase: str, game_files: Sequence[str], epoch: int
) -> Any:
    from alfworld.agents.environment import get_environment

    wrapper = get_environment("AlfredTWEnv")(
        load_alfworld_config(config), train_eval=SPLIT_NAMES[phase]
    )
    wrapper.game_files = list(game_files)
    wrapper.num_games = len(wrapper.game_files)
    env = wrapper.init_env(batch_size=1)
    env.seed(int(config["experiment"]["seed"]) + epoch)
    # Pinned TextWorld 1.6.2 reshuffles its game iterator inside seed(). The
    # manifest already fixes sample selection AND execution order, so enforce
    # that order and stop at its end instead of cycling into another task.
    if not hasattr(env, "_gamefiles_iterator"):
        env.close()
        raise RuntimeError("Pinned TextWorld ordered-game iterator is unavailable")
    env._gamefiles_iterator = iter(wrapper.game_files)
    return env


def run_method(
    method_name: str,
    config: Dict[str, Any],
    run_root: Path,
    manifest: Dict[str, List[str]],
    limit: Optional[int] = None,
) -> Path:
    method_root = run_root / method_name
    method_root.mkdir(parents=True, exist_ok=True)
    trajectory_path = method_root / "trajectories.jsonl"
    state_path = method_root / "method_state.json"
    model_config = config["model"]
    if not model_config.get("name") or model_config["name"] == "auto":
        raise ValueError(
            "Use an explicit model.name so resume can freeze model identity"
        )
    for phase, games in manifest.items():
        if len(games) != len(set(games)):
            raise ValueError("Duplicate game files in phase: " + phase)
    data_files = set(Path(game) for games in manifest.values() for game in games)
    for game in list(data_files):
        data_files.update(
            p
            for p in game.parent.iterdir()
            if p.suffix in {".json", ".pddl", ".tw-pddl", ".twl2"}
        )
    data_root = Path(config["environment"]["data_root"])
    data_files.update(
        [
            data_root / "logic" / "alfred.pddl",
            data_root / "logic" / "alfred.twl2",
            Path(config["environment"]["alfworld_config"]),
        ]
    )
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_files)
    }
    guard = RunGuard(
        method_root,
        {
            "config": redacted_config(config),
            "stream": manifest,
            "data": hashes,
            "implementation": source_identity(Path(__file__).resolve().parents[1]),
        },
    )
    store = CheckpointStore(method_root)
    if store.method is not None:
        atomic_json(state_path, store.method)
    client = OpenAICompatibleClient(
        base_url=str(model_config["base_url"]),
        api_key=resolve_api_key(model_config),
        model=str(model_config.get("name", "auto")),
        timeout=int(model_config.get("timeout_seconds", 180)),
        retries=int(model_config.get("retries", 3)),
        extra_body=model_config.get("extra_body"),
    )

    def auxiliary(prompt: str, purpose: str) -> str:
        return client.complete(
            [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            purpose=purpose,
            temperature=float(model_config.get("temperature", 0.0)),
            top_p=float(model_config.get("top_p", 1.0)),
            max_tokens=int(model_config.get("auxiliary_max_tokens", 512)),
        )

    method = create_method(method_name, config, state_path, auxiliary)
    completed = {item["episode_id"]: item for item in store.records}
    planned = [
        "{}-e{}-{:05d}".format(phase, epoch, i)
        for phase in ("train", "valid_seen", "valid_unseen")
        for epoch in range(
            int(config["phases"][phase].get("epochs", 1)) if phase == "train" else 1
        )
        for i in range(len(manifest.get(phase, [])))
    ]
    if list(completed) != planned[: len(completed)] or len(completed) > len(planned):
        raise RuntimeError("Checkpoint is not the exact ordered episode prefix")
    guard.recover(completed)
    store.export_jsonl(trajectory_path)
    client.journal_path = method_root / "request_events.jsonl"
    update_phases = memory_update_phases(config)
    completed_updates = sum(
        item.get("phase") in update_phases for item in completed.values()
    )
    if method.step != completed_updates:
        raise RuntimeError(
            "resume safety check failed for {}: state has {} memory updates but "
            "trajectory has {} completed update-phase episodes".format(
                method_name, method.step, completed_updates
            )
        )
    resolved_model = (
        client.model if len(completed) == len(planned) else client.resolve_model()
    )
    print(
        "\n=== method={} model={} ===".format(method_name, resolved_model), flush=True
    )
    processed = 0
    limit_reached = False
    for phase in ("train", "valid_seen", "valid_unseen"):
        if limit_reached or len(completed) == len(planned):
            break
        if not manifest.get(phase):
            continue
        phase_epochs = (
            int(config["phases"][phase].get("epochs", 1)) if phase == "train" else 1
        )
        for epoch in range(phase_epochs):
            if limit_reached:
                break
            env = create_env(config, phase, manifest[phase], epoch)
            try:
                for episode_index in range(len(manifest[phase])):
                    if limit is not None and processed >= limit:
                        limit_reached = True
                        break
                    episode_id = "{}-e{}-{:05d}".format(phase, epoch, episode_index)
                    if episode_id in completed:
                        env.reset()  # advance deterministic TextWorld game iterator
                        continue
                    guard.start(episode_id)
                    client.active_task_key = episode_id
                    result, _ = run_episode(
                        env,
                        client,
                        method,
                        config,
                        phase,
                        epoch,
                        episode_index,
                        expected_gamefile=manifest[phase][episode_index],
                    )
                    processed += 1
                    result.update(
                        {
                            "method": method.name,
                            "implementation_mode": method.implementation_mode,
                            "model": resolved_model,
                        }
                    )
                    payload = {
                        "method": method.name,
                        "implementation_mode": method.implementation_mode,
                        "step": method.step,
                        "state": method.state,
                        "auxiliary_parse_errors": method.auxiliary_parse_errors,
                    }
                    store.commit(payload, result)
                    method.save()
                    store.export_jsonl(trajectory_path)
                    guard.finish(episode_id)
                    completed[episode_id] = result
                    print(
                        "{} e{} {}/{} {} steps={} sr={:.3f}".format(
                            phase,
                            epoch,
                            episode_index + 1,
                            len(manifest[phase]),
                            "SUCCESS" if result["success"] else "FAIL",
                            result["steps"],
                            sum(
                                item["success"]
                                for item in completed.values()
                                if item["phase"] == phase
                            )
                            / max(
                                1,
                                sum(
                                    item["phase"] == phase
                                    for item in completed.values()
                                ),
                            ),
                        ),
                        flush=True,
                    )
            finally:
                env.close()
    summary_path = method_root / "summary.json"
    summary = summarize_results(
        list(load_completed(trajectory_path).values()),
        method.summary(),
        resolved_model,
        config,
        run_root / "game_manifest.json",
    )
    summary["completed_episodes"] = len(completed)
    summary["planned_episodes"] = len(planned)
    summary["is_complete"] = len(completed) == len(planned)
    summary["checkpoint_protocol"] = "appendix-paper-v2"
    atomic_json(summary_path, summary)
    print(json.dumps(summary["phases"], ensure_ascii=False, indent=2), flush=True)
    return summary_path


def summarize_results(
    results: Sequence[Dict[str, Any]],
    method_summary: Dict[str, Any],
    model: str,
    config: Dict[str, Any],
    manifest_path: Path,
) -> Dict[str, Any]:
    phases: Dict[str, Dict[str, Any]] = {}
    for phase in ("train", "valid_seen", "valid_unseen"):
        subset = [item for item in results if item["phase"] == phase]
        if not subset:
            continue
        task_types: Dict[str, Dict[str, Any]] = {}
        for task_type in sorted({item["task_type"] for item in subset}):
            typed = [item for item in subset if item["task_type"] == task_type]
            task_types[task_type] = {
                "episodes": len(typed),
                "successes": sum(item["success"] for item in typed),
                "success_rate": sum(item["success"] for item in typed) / len(typed),
            }
        llm_totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for item in subset:
            for purpose, values in (item.get("llm") or {}).items():
                for key, value in values.items():
                    llm_totals[purpose][key] += float(value)
        phases[phase] = {
            "episodes": len(subset),
            "successes": sum(item["success"] for item in subset),
            "success_rate": sum(item["success"] for item in subset) / len(subset),
            "average_steps": sum(int(item["steps"]) for item in subset) / len(subset),
            "invalid_actions": sum(int(item["invalid_actions"]) for item in subset),
            "repair_calls": sum(int(item["repair_calls"]) for item in subset),
            "wall_seconds": sum(float(item["wall_seconds"]) for item in subset),
            "by_task_type": task_types,
            "llm": {purpose: dict(values) for purpose, values in llm_totals.items()},
        }
    protocol_config = copy.deepcopy(config)
    protocol_config = redacted_config(protocol_config)
    config_hash = hashlib.sha256(
        json.dumps(protocol_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "method": method_summary,
        "model": model,
        "api_base": config["model"]["base_url"],
        "protocol": (
            "memory updates after each episode in: "
            + ", ".join(memory_update_phases(config))
        ),
        "memory_update_phases": list(memory_update_phases(config)),
        "expert_trajectory_access": False,
        "action_space": "admissible_commands",
        "config_sha256": config_hash,
        "manifest_path": str(manifest_path),
        "phases": phases,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--methods",
        default="all",
        help="comma-separated methods or all ({})".format(",".join(METHODS)),
    )
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--limit",
        type=int,
        help="run at most this many new episodes per selected method",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if args.api_base:
        config["model"]["base_url"] = args.api_base
    if args.model:
        config["model"]["name"] = args.model
    if args.output_root:
        config["experiment"]["output_root"] = args.output_root
    resolve_model_environment(config)
    resolve_environment_paths(config)
    names = (
        list(METHODS)
        if args.methods == "all"
        else [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    )
    unknown = [name for name in names if name not in METHODS]
    if unknown:
        raise ValueError("unknown methods: " + ", ".join(unknown))
    run_root = Path(
        str(config["experiment"]["output_root"])
    ).expanduser().resolve() / str(config["experiment"]["run_name"])
    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_root / "resolved_config.yaml"
    protocol_config = copy.deepcopy(config)
    protocol_config = redacted_config(protocol_config)
    if resolved_config_path.exists():
        prior = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
        # Backward-compatible normalization for runs created before endpoints
        # were treated as per-method execution resources.
        prior_for_comparison = copy.deepcopy(prior)
        prior_for_comparison = redacted_config(prior_for_comparison)
        if prior_for_comparison != protocol_config:
            raise RuntimeError(
                "run directory already has a different config; choose another run_name or output_root"
            )
    else:
        resolved_config_path.write_text(
            yaml.safe_dump(protocol_config, sort_keys=False), encoding="utf-8"
        )
    manifest = prepare_manifest(config, run_root)
    summaries = []
    for name in names:
        summaries.append(str(run_method(name, config, run_root, manifest, args.limit)))
    print("completed summaries:\n" + "\n".join(summaries), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from continual_alfworld.methods import EpisodeOutcome, create_method
from continual_alfworld.runner import run_method, create_env as real_create_env
from tokenworth_core.runtime import CheckpointStore


def config(root):
    (root / "logic").mkdir(exist_ok=True)
    for name in ("alfred.pddl", "alfred.twl2"):
        (root / "logic" / name).write_text("fixture")
    (root / "base.yaml").write_text("fixture")
    return {
        "experiment": {"seed": 42},
        "environment": {
            "data_root": str(root),
            "alfworld_config": str(root / "base.yaml"),
        },
        "phases": {"train": {"epochs": 1}, "valid_seen": {}, "valid_unseen": {}},
        "retrieval": {"top_k": 1, "dimensions": 8},
        "memory": {
            "trajectory_format": "state_delta",
            "max_records": 4,
            "max_item_chars": 4096,
            "context_chars": 4200,
            "update_phases": ["train", "valid_seen", "valid_unseen"],
        },
        "methods": {"tokenworth": {}},
        "model": {
            "base_url": "http://example.invalid",
            "api_key": "test",
            "name": "fixed-model",
            "action_max_tokens": 128,
        },
        "agent": {
            "max_steps": 2,
            "repair_attempts": 1,
            "history_turns": 12,
            "observation_chars": 1800,
            "memory_prompt_format": "method_reference",
        },
    }


def outcome(success):
    return EpisodeOutcome(
        "train",
        0,
        "fixture",
        "put an apple in the fridge",
        "pick_and_place_simple",
        "fixture",
        success,
        1,
        [
            {
                "observation": "You see an apple.",
                "action": "look",
                "next_observation": "done",
            }
        ],
    )


class Client:
    calls = 0
    fail_at = None

    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.stats = {}

    def resolve_model(self):
        return self.model

    def complete(self, *args, **kwargs):
        type(self).calls += 1
        if self.fail_at == type(self).calls:
            raise RuntimeError("service fault")
        value = self.stats.setdefault(
            "action",
            dict(
                calls=0,
                prompt_tokens=0,
                completion_tokens=0,
                latency_seconds=0,
                errors=0,
            ),
        )
        value["calls"] += 1
        value["prompt_tokens"] += 3
        value["completion_tokens"] += 2
        return '{"action":"look"}'

    def snapshot_stats(self):
        return copy.deepcopy(self.stats)


class Env:
    def __init__(self, files):
        self.files = files
        self.index = -1

    def reset(self):
        self.index += 1
        return ["Your task is to: put an apple in the fridge"], {
            "extra.gamefile": [self.files[self.index]],
            "admissible_commands": [["look"]],
        }

    def step(self, actions):
        return ["done"], [1], [True], {"won": [True], "admissible_commands": [["look"]]}

    def close(self):
        pass


class PaperTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = config(self.root)
        files = []
        for i in range(2):
            path = (
                self.root
                / ("pick_and_place_simple-Apple-None-Fridge-{}".format(i))
                / "trial"
                / "game.tw-pddl"
            )
            path.parent.mkdir(parents=True)
            path.write_text("fixture-{}".format(i))
            files.append(str(path))
        self.manifest = {"train": files, "valid_seen": [], "valid_unseen": []}
        Client.calls = 0
        Client.fail_at = None
        for name, value in (
            ("OpenAICompatibleClient", Client),
            ("create_env", lambda config, phase, files, epoch: Env(files)),
        ):
            p = patch("continual_alfworld.runner." + name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch("builtins.print")
        p.start()
        self.addCleanup(p.stop)

    def execute(self, limit=None):
        return run_method(
            "tokenworth",
            copy.deepcopy(self.config),
            self.root / "run",
            self.manifest,
            limit,
        )

    def test_partial_and_completed_resume(self):
        path = self.execute(1)
        self.assertFalse(json.loads(path.read_text())["is_complete"])
        self.execute()
        self.assertEqual(Client.calls, 2)
        store = CheckpointStore(self.root / "run" / "tokenworth")
        self.assertEqual(store.method["state"]["paper"]["metrics"]["pair_writes"], 1)
        self.assertTrue(json.loads(path.read_text())["is_complete"])
        self.execute()
        self.assertEqual(Client.calls, 2)

    def test_committed_crash_no_replay(self):
        original = CheckpointStore.commit

        def interrupted(store, method, record):
            original(store, method, record)
            raise RuntimeError("after commit")

        with (
            patch.object(CheckpointStore, "commit", interrupted),
            self.assertRaisesRegex(RuntimeError, "after commit"),
        ):
            self.execute(1)
        self.execute()
        self.assertEqual(Client.calls, 2)

    def test_uncommitted_failure_refuses_replay(self):
        self.execute(1)
        Client.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "service fault"):
            self.execute()
        with self.assertRaisesRegex(RuntimeError, "replay refused"):
            self.execute()
        self.assertEqual(Client.calls, 2)
        self.assertEqual(
            len(CheckpointStore(self.root / "run" / "tokenworth").records), 1
        )

    def test_model_and_data_drift_rejected(self):
        self.execute(1)
        self.config["model"]["base_url"] = "http://different.invalid"
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.execute()
        self.assertEqual(Client.calls, 1)
        self.config["model"]["base_url"] = "http://example.invalid"
        Path(self.manifest["train"][0]).write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.execute()
        self.assertEqual(Client.calls, 1)

    def test_frozen_phases_do_not_learn(self):
        self.config["memory"]["update_phases"] = ["train"]
        files = self.manifest["train"]
        self.manifest = {
            "train": files[:1],
            "valid_seen": files[1:],
            "valid_unseen": files[:1],
        }
        self.execute()
        store = CheckpointStore(self.root / "run" / "tokenworth")
        self.assertEqual(len(store.records), 3)
        self.assertEqual(store.method["step"], 1)
        self.assertEqual(store.method["state"]["paper"]["metrics"]["pair_writes"], 0)

    def test_cached_residual_and_state_round_trip_without_auxiliary(self):
        def forbidden(*args):
            raise AssertionError("No auxiliary LLM")

        path = self.root / "state.json"
        method = create_method("tokenworth", self.config, path, forbidden)
        method.render_context(method.retrieve(outcome(True).task, "type"))
        method.observe(outcome(True))
        items = method.retrieve(outcome(False).task, "type")
        cached = copy.deepcopy(method.core._selected[0])
        self.assertLessEqual(len(method.render_context(items)), 4200)
        failed = outcome(False)
        failed.memory_disabled_after_loop = True
        failed.loop_break_forced_actions = 1
        method.observe(failed)
        method.save()
        pair = method.core.state["pairs"][0]
        self.assertEqual(pair["feature"], cached["feature"])
        self.assertAlmostEqual(pair["label"], -1 - math.tanh(cached["base_score"] / 2))
        self.assertEqual(method.core.state["base_pairs"][-1]["label"], -1)
        restored = create_method("tokenworth", self.config, path, forbidden)
        self.assertEqual(restored.core.state, method.core.state)

    def test_legacy_gates_and_state_rejected(self):
        path = self.root / "state.json"
        for key in (
            "need_memory_gate_enabled",
            "structured_gate_enabled",
            "feedback_credit_mode",
            "eviction_rfm_weight",
        ):
            self.config["methods"]["tokenworth"] = {key: True}
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "Unknown paper"),
            ):
                create_method("tokenworth", self.config, path, lambda *args: "")
        self.config["methods"]["tokenworth"] = {}
        path.write_text(
            json.dumps({"method": "tokenworth", "step": 1, "state": {"memories": []}})
        )
        with self.assertRaisesRegex(ValueError, "Legacy"):
            create_method("tokenworth", self.config, path, lambda *args: "")

    def test_auto_model_and_duplicate_manifest_are_rejected(self):
        self.config["model"]["name"] = "auto"
        with self.assertRaisesRegex(ValueError, "explicit model"):
            self.execute()
        self.config["model"]["name"] = "fixed-model"
        self.manifest["train"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate game"):
            self.execute()

    def test_environment_order_mismatch_cannot_commit_feedback(self):
        files = list(reversed(self.manifest["train"]))
        with patch("continual_alfworld.runner.create_env", lambda *args: Env(files)):
            with self.assertRaisesRegex(RuntimeError, "game order"):
                self.execute()
        self.assertFalse(
            (self.root / "run" / "tokenworth" / "checkpoint.json").exists()
        )

    def test_pinned_engine_seed_cannot_shuffle_manifest(self):
        from types import ModuleType
        import sys

        class ShufflingEngine:
            def seed(self, value):
                self._gamefiles_iterator = iter(reversed(["first", "second"]))

            def close(self):
                pass

        engine = ShufflingEngine()
        engine.seed(0)

        class Wrapper:
            def __init__(self, *args, **kwargs):
                pass

            def init_env(self, batch_size):
                return engine

        module = ModuleType("alfworld.agents.environment")
        module.get_environment = lambda name: Wrapper
        with patch.dict(sys.modules, {"alfworld.agents.environment": module}), patch(
            "continual_alfworld.runner.load_alfworld_config", return_value={}
        ):
            created = real_create_env(self.config, "train", ["first", "second"], 0)
        self.assertEqual(list(created._gamefiles_iterator), ["first", "second"])

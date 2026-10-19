import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from continual_alfworld.runner import (
    METHOD_REFERENCE_SYSTEM_PROMPT,
    extract_action,
    format_turn,
    is_memory_update_phase,
    load_config,
    resolve_api_key,
    resolve_environment_paths,
    resolve_model_environment,
    memory_update_phases,
    parse_args,
    select_games,
    task_type_from_gamefile,
)


def game(prefix, index):
    return "fixtures/games/{0}-Object-None-Receptacle-{1}/trial/game.tw-pddl".format(prefix, index)


class RunnerTests(unittest.TestCase):
    def test_episode_limit_must_be_positive(self):
        self.assertEqual(parse_args(["--config", "config.yaml", "--limit", "1"]).limit, 1)
        with self.assertRaises(SystemExit):
            parse_args(["--config", "config.yaml", "--limit", "0"])

    def test_cloud_model_environment_keeps_api_key_out_of_resolved_config(self):
        config = {
            "model": {
                "base_url": "",
                "base_url_env": "ALFWORLD_TEST_BASE",
                "name": "",
                "name_env": "ALFWORLD_TEST_MODEL",
                "api_key": "",
                "api_key_env": "ALFWORLD_TEST_KEY",
            }
        }
        with patch.dict(
            "os.environ",
            {
                "ALFWORLD_TEST_BASE": "https://example.invalid",
                "ALFWORLD_TEST_MODEL": "example-model",
                "ALFWORLD_TEST_KEY": "test-secret",
            },
        ):
            resolve_model_environment(config)
            self.assertEqual(resolve_api_key(config["model"]), "test-secret")
        self.assertEqual(config["model"]["base_url"], "https://example.invalid")
        self.assertEqual(config["model"]["name"], "example-model")
        self.assertNotIn("test-secret", str(config))

    def test_cloud_model_environment_rejects_missing_credentials(self):
        config = {"model": {"api_key_env": "ALFWORLD_TEST_MISSING_KEY"}}
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "ALFWORLD_TEST_MISSING_KEY"):
                resolve_model_environment(config)

    def test_cloud_model_environment_reads_env_file_without_persisting_key(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "ALFWORLD_TEST_BASE='https://example.invalid/v1'\n"
                "export ALFWORLD_TEST_MODEL=example-model\n"
                "ALFWORLD_TEST_KEY=from-file-secret\n",
                encoding="utf-8",
            )
            config = {
                "model": {
                    "base_url_env": "ALFWORLD_TEST_BASE",
                    "name_env": "ALFWORLD_TEST_MODEL",
                    "api_key_env": "ALFWORLD_TEST_KEY",
                    "env_file": str(env_path),
                }
            }
            with patch.dict("os.environ", {}, clear=True):
                resolve_model_environment(config)
                self.assertEqual(resolve_api_key(config["model"]), "from-file-secret")
            self.assertEqual(config["model"]["base_url"], "https://example.invalid/v1")
            self.assertEqual(config["model"]["name"], "example-model")
            self.assertNotIn("from-file-secret", str(config))

    def test_environment_paths_resolve_from_model_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            base_config = Path(directory) / "alfworld" / "base_config.yaml"
            data_root = Path(directory) / "alfworld-data"
            env_path.write_text(
                "ALFWORLD_TEST_CONFIG={}\nALFWORLD_TEST_DATA={}\n".format(
                    base_config, data_root
                ),
                encoding="utf-8",
            )
            config = {
                "model": {"env_file": str(env_path)},
                "environment": {
                    "alfworld_config": "",
                    "alfworld_config_env": "ALFWORLD_TEST_CONFIG",
                    "data_root": "",
                    "data_root_env": "ALFWORLD_TEST_DATA",
                },
            }
            with patch.dict("os.environ", {}, clear=True):
                resolve_environment_paths(config)
            self.assertEqual(
                config["environment"]["alfworld_config"],
                str(base_config),
            )
            self.assertEqual(
                config["environment"]["data_root"], str(data_root)
            )

    def test_memory_updates_default_to_train_only(self):
        config = {"memory": {}}
        self.assertEqual(memory_update_phases(config), ("train",))
        self.assertTrue(is_memory_update_phase(config, "train"))
        self.assertFalse(is_memory_update_phase(config, "valid_seen"))

    def test_memory_updates_can_continue_through_evaluation(self):
        config = {
            "memory": {
                "update_phases": ["train", "valid_seen", "valid_unseen"],
            }
        }
        self.assertTrue(is_memory_update_phase(config, "valid_seen"))
        self.assertTrue(is_memory_update_phase(config, "valid_unseen"))

    def test_memory_update_phases_reject_unknown_phase(self):
        with self.assertRaisesRegex(ValueError, "unknown memory.update_phases"):
            memory_update_phases({"memory": {"update_phases": ["train", "test"]}})

    def test_config_variant_deep_merges_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "experiment:\n  seed: 42\n  run_name: base\nmethods:\n  tokenworth:\n    threshold: 0.5\n",
                encoding="utf-8",
            )
            (root / "variant.yaml").write_text(
                "extends: base.yaml\nexperiment:\n  run_name: variant\nmethods:\n  tokenworth:\n    threshold: 0.75\n",
                encoding="utf-8",
            )
            config = load_config(root / "variant.yaml")
            self.assertEqual(config["experiment"], {"seed": 42, "run_name": "variant"})
            self.assertEqual(config["methods"]["tokenworth"]["threshold"], 0.75)

    def test_memory_prompt_marks_past_state_as_non_authoritative(self):
        turn = format_turn(
            "put an apple in the fridge",
            "You are in a kitchen.",
            ["look", "go to fridge 1"],
            [],
            "Past episode initial state: apple 7 was in cabinet 4",
            12,
            1800,
        )
        combined = METHOD_REFERENCE_SYSTEM_PROMPT + "\n" + turn
        self.assertIn("completed past episode", METHOD_REFERENCE_SYSTEM_PROMPT)
        self.assertIn("object numbers, locations, layouts", combined)
        self.assertIn("Do not reuse any of its task goals", turn)
        self.assertIn("admissible actions", turn)

    def test_method_reference_orders_past_boundary_then_current_task(self):
        task = "examine the keychain with the desklamp"
        memory = "Past task: examine the pillow with the desklamp"
        turn = format_turn(
            task,
            "You see a keychain 3 on dresser 1.",
            ["take keychain 3 from dresser 1"],
            [],
            memory,
            12,
            1800,
            "method_reference",
        )
        self.assertIn("METHOD reference", METHOD_REFERENCE_SYSTEM_PROMPT)
        self.assertIn("Never use a substitute or placeholder object", METHOD_REFERENCE_SYSTEM_PROMPT)
        self.assertLess(turn.index(memory), turn.index("END OF PAST EPISODE"))
        self.assertLess(turn.index("END OF PAST EPISODE"), turn.index(task))
        self.assertLess(turn.index(task), turn.index("Current observation:"))
        self.assertIn("CURRENT episode and are authoritative", turn)

    def test_method_reference_rejects_unknown_prompt_format(self):
        with self.assertRaisesRegex(ValueError, "memory_prompt_format"):
            format_turn("task", "observation", ["look"], [], "(none)", 12, 1800, "unknown")

    def test_action_parser_exact_and_fallback(self):
        actions = ["look", "go to fridge 1", "open fridge 1"]
        action, _, valid = extract_action('{"reasoning":"inspect","action":"open fridge 1"}', actions)
        self.assertEqual(action, "open fridge 1")
        self.assertTrue(valid)
        action, _, valid = extract_action("nonsense", actions)
        self.assertEqual(action, "look")
        self.assertFalse(valid)

    def test_stratified_manifest_is_deterministic_and_balanced(self):
        prefixes = [
            "pick_and_place_simple",
            "look_at_obj_in_light",
            "pick_clean_then_place_in_recep",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_two_obj_and_place",
        ]
        files = [game(prefix, index) for prefix in prefixes for index in range(4)]
        first = select_games(files, 12, "stratified", 42)
        second = select_games(list(reversed(files)), 12, "stratified", 42)
        self.assertEqual(first, second)
        counts = {prefix: 0 for prefix in prefixes}
        for item in first:
            counts[task_type_from_gamefile(item)] += 1
        self.assertEqual(set(counts.values()), {2})


if __name__ == "__main__":
    unittest.main()

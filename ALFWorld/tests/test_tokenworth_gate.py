import unittest

from continual_alfworld.tokenworth_gate import (
    CompatibilityConfig,
    LightweightAdvantageGate,
    TaskTypeBaseline,
    compatible_candidate,
    deterministic_probe,
    filter_compatible_candidates,
    parse_alfworld_task,
    signature_from_candidate,
)


class TaskParsingTests(unittest.TestCase):
    def test_parses_all_six_task_families(self):
        cases = [
            ("put an apple in the fridge", "pick_and_place_simple", "place", "apple", "fridge"),
            ("clean some lettuce and put it in diningtable", "pick_clean_then_place_in_recep", "clean", "lettuce", "diningtable"),
            ("heat an egg and put it on countertop", "pick_heat_then_place_in_recep", "heat", "egg", "countertop"),
            ("cool a bowl and put it in cabinet", "pick_cool_then_place_in_recep", "cool", "bowl", "cabinet"),
            ("put two cellphones in sofa", "pick_two_obj_and_place", "two_object", "cellphones", "sofa"),
            ("examine the keychain with the desk lamp", "look_at_obj_in_light", "light", "keychain", "desklamp"),
        ]
        for task, task_type, operation, target, destination in cases:
            with self.subTest(task_type=task_type):
                signature = parse_alfworld_task(task, task_type)
                self.assertEqual(signature.task_type, task_type)
                self.assertEqual(signature.operation, operation)
                self.assertEqual(signature.target, target)
                self.assertEqual(signature.destination, destination)

    def test_parses_adjectival_state_change(self):
        signature = parse_alfworld_task("put a cool cup in microwave")
        self.assertEqual((signature.operation, signature.target, signature.destination), ("cool", "cup", "microwave"))

    def test_candidate_falls_back_to_trace_task_header(self):
        signature = signature_from_candidate(
            {
                "task_type": "pick_and_place_simple",
                "content": "Task: put an apple in the fridge\nOutcome: success",
            }
        )
        self.assertEqual(signature.target, "apple")
        self.assertEqual(signature.destination, "fridge")


class CompatibilityTests(unittest.TestCase):
    def test_rejects_same_destination_but_wrong_operation(self):
        current = parse_alfworld_task("put a cool cup in microwave", "pick_cool_then_place_in_recep")
        old = parse_alfworld_task("put bread in microwave", "pick_and_place_simple")
        self.assertFalse(compatible_candidate(current, old))

    def test_rejects_wrong_target_even_with_same_task_type(self):
        current = parse_alfworld_task("examine the keychain with the desklamp", "look_at_obj_in_light")
        old = parse_alfworld_task("examine the pillow with the desklamp", "look_at_obj_in_light")
        self.assertFalse(compatible_candidate(current, old))

    def test_individual_hard_filters_are_configurable(self):
        current = parse_alfworld_task("put a cool cup in microwave", "pick_cool_then_place_in_recep")
        old = parse_alfworld_task("put bread in microwave", "pick_and_place_simple")
        relaxed = CompatibilityConfig(
            exact_task_type=False, exact_operation=False, exact_target=False
        )
        self.assertTrue(compatible_candidate(current, old, relaxed))

    def test_filters_existing_tokenworth_records(self):
        records = [
            {"id": "good", "query": "put a cool bowl in diningtable", "task_type": "pick_cool_then_place_in_recep"},
            {"id": "wrong-operation", "query": "put a bowl in cabinet", "task_type": "pick_and_place_simple"},
            {"id": "wrong-target", "query": "put a cool apple in cabinet", "task_type": "pick_cool_then_place_in_recep"},
        ]
        selected = filter_compatible_candidates(
            "put a cool bowl in cabinet", "pick_cool_then_place_in_recep", records
        )
        self.assertEqual([item["id"] for item in selected], ["good"])


class ProbeAndAdvantageTests(unittest.TestCase):
    def test_probe_is_deterministic_and_seeded(self):
        first = [deterministic_probe(42, "episode-{}".format(i), "put apple in fridge", 0.3) for i in range(100)]
        second = [deterministic_probe(42, "episode-{}".format(i), "put apple in fridge", 0.3) for i in range(100)]
        other_seed = [deterministic_probe(43, "episode-{}".format(i), "put apple in fridge", 0.3) for i in range(100)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_seed)
        self.assertGreater(sum(first), 15)
        self.assertLess(sum(first), 45)
        self.assertFalse(deterministic_probe(42, "x", "task", 0.0))
        self.assertTrue(deterministic_probe(42, "x", "task", 1.0))

    def test_baseline_tracks_task_types_independently_and_round_trips(self):
        baseline = TaskTypeBaseline(ema_alpha=0.5, beta_weight=0.5)
        baseline.update("clean", True)
        baseline.update("clean", True)
        baseline.update("cool", False)
        self.assertGreater(baseline.estimate("clean"), baseline.estimate("cool"))
        restored = TaskTypeBaseline(baseline.to_state(), ema_alpha=0.5, beta_weight=0.5)
        self.assertAlmostEqual(restored.estimate("clean"), baseline.estimate("clean"))
        self.assertEqual(restored.count("clean"), 2)

    def test_advantage_compares_memory_against_no_memory_baseline(self):
        baseline = TaskTypeBaseline(beta_weight=1.0)
        for success in (True, True, True, False):
            baseline.update("pick_cool_then_place_in_recep", success)
        gate = LightweightAdvantageGate(
            baseline, probe_rate=0.0, advantage_threshold=0.03,
            min_baseline_feedback=4, min_memory_feedback=4, model_weight=0.0,
        )
        weak = {"help_count": 1, "harm_count": 3}
        strong = {"help_count": 4, "harm_count": 0}
        self.assertLess(gate.predicted_advantage("pick_cool_then_place_in_recep", weak), 0.0)
        self.assertGreater(gate.predicted_advantage("pick_cool_then_place_in_recep", strong), 0.03)

    def test_gate_constructs_from_yaml_style_mapping(self):
        gate = LightweightAdvantageGate.from_mapping(
            {
                "gate_probe_rate": 0.1,
                "gate_advantage_threshold": 0.07,
                "gate_min_feedback": 6,
                "gate_model_weight": 0.0,
            },
            {
                "pick_cool_then_place_in_recep": {
                    "alpha": 4.0, "beta": 2.0, "ema": 0.6, "count": 4,
                }
            },
        )
        self.assertEqual(gate.probe_rate, 0.1)
        self.assertEqual(gate.advantage_threshold, 0.07)
        self.assertEqual(gate.min_baseline_feedback, 6)
        self.assertEqual(gate.baseline.count("pick_cool_then_place_in_recep"), 4)

    def test_decision_prioritizes_compatibility_then_probe_then_advantage(self):
        baseline = TaskTypeBaseline(beta_weight=1.0)
        for success in (True, False, True, False):
            baseline.update("pick_cool_then_place_in_recep", success)
        gate = LightweightAdvantageGate(
            baseline, probe_rate=1.0, advantage_threshold=0.03,
            min_baseline_feedback=4, min_memory_feedback=4, model_weight=0.0,
        )
        wrong = {
            "query": "put bread in microwave",
            "task_type": "pick_and_place_simple",
            "help_count": 9,
            "harm_count": 0,
        }
        rejected = gate.decide(
            "put a cool cup in microwave", "pick_cool_then_place_in_recep", wrong,
            learning=True, seed=42, episode_key="e1",
        )
        self.assertEqual(rejected.reason, "incompatible")
        self.assertFalse(rejected.no_memory_probe)

        compatible = {
            "query": "put a cool cup in cabinet",
            "task_type": "pick_cool_then_place_in_recep",
            "help_count": 4,
            "harm_count": 0,
        }
        probe = gate.decide(
            "put a cool cup in microwave", "pick_cool_then_place_in_recep", compatible,
            learning=True, seed=42, episode_key="e2",
        )
        self.assertEqual(probe.reason, "no_memory_probe")
        self.assertTrue(probe.no_memory_probe)

        evaluation = gate.decide(
            "put a cool cup in microwave", "pick_cool_then_place_in_recep", compatible,
            learning=False, seed=42, episode_key="e2",
        )
        self.assertTrue(evaluation.use_memory)
        self.assertEqual(evaluation.reason, "positive_advantage")

    def test_evaluation_abstains_on_unvalidated_memory_by_default(self):
        gate = LightweightAdvantageGate(
            TaskTypeBaseline(), probe_rate=0.0, min_baseline_feedback=4,
            min_memory_feedback=4, model_weight=0.0,
        )
        candidate = {
            "query": "put a cool bowl in cabinet",
            "task_type": "pick_cool_then_place_in_recep",
            "help_count": 0,
            "harm_count": 0,
        }
        decision = gate.decide(
            "put a cool bowl in microwave", "pick_cool_then_place_in_recep", candidate,
            learning=False, seed=42, episode_key="eval-1",
        )
        self.assertFalse(decision.use_memory)
        self.assertEqual(decision.reason, "baseline_warmup")


if __name__ == "__main__":
    unittest.main()

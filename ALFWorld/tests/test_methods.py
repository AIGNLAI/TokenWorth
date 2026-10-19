import copy
import json
import tempfile
import unittest
from pathlib import Path

from continual_alfworld.methods import EpisodeOutcome, RetrievedMemory, create_method
from continual_alfworld.trajectory import build_state_delta_trace


def fake_auxiliary(prompt, purpose):
    del prompt
    responses = {
        "memory_mem0": {
            "operations": [
                {
                    "op": "add",
                    "target_id": "",
                    "content": "Open a closed receptacle before taking its object.",
                    "tags": ["open"],
                }
            ]
        },
        "memory_memp_success": {
            "detailed_steps": [
                "Take the target, then place it in the named receptacle."
            ],
            "abstraction": "Satisfy manipulation preconditions before placement.",
            "tags": ["place"],
        },
        "memory_memp_failure": {
            "cautions": ["Do not search the same empty receptacle repeatedly."],
            "revisions": [],
            "deprecate": [],
        },
        "memory_memrl": {
            "content": "Track searched receptacles and change location after an empty search.",
            "tags": ["search"],
        },
        "memory_reflexion": {
            "failure_pattern": "looping",
            "root_cause": "no search bookkeeping",
            "corrective_plan": "visit each receptacle once",
            "checks": ["check inventory"],
        },
        "memory_memskill_executor": {
            "operations": [
                {
                    "skill_id": "skill-insert",
                    "op": "insert",
                    "target_id": "",
                    "content": "Record which receptacles were already searched.",
                    "tags": ["search"],
                }
            ]
        },
        "memory_memskill_designer": {
            "name": "merge duplicates",
            "triggers": ["duplicate"],
            "instruction": "Merge semantically duplicate search memories.",
        },
        "memory_evoskill_proposer": {
            "action": "create",
            "target_id": "",
            "rationale": "search failures",
            "skill_brief": "systematic search",
        },
        "memory_evoskill_builder": {
            "name": "systematic search",
            "triggers": ["find", "object"],
            "instruction": "Search unvisited receptacles in a fixed order.",
        },
        "memory_evosc_extract": {
            "content": "Use a fixed order when searching receptacles.",
            "tags": ["search"],
        },
        "memory_evosc_consolidate": {
            "summary": "Search each receptacle once; open it first when closed.",
            "tags": ["search", "open"],
        },
    }
    return json.dumps(responses[purpose])


def outcome(success):
    return EpisodeOutcome(
        phase="train",
        epoch=0,
        episode_id="train-e0-00000",
        task="put an apple in the fridge",
        task_type="pick_and_place_simple",
        gamefile="/x/pick_and_place_simple-Apple-None-Fridge-1/trial/game.tw-pddl",
        success=success,
        steps=2,
        transcript=[
            {"observation": "You see a fridge.", "action": "go to fridge 1"},
            {"observation": "The fridge is closed.", "action": "open fridge 1"},
        ],
    )


def delta_outcome(success=True):
    return EpisodeOutcome(
        phase="train",
        epoch=0,
        episode_id="train-e0-00001",
        task="clean some lettuce and put it in diningtable",
        task_type="pick_clean_then_place_in_recep",
        gamefile="/x/pick_clean_then_place_in_recep-Lettuce-Sink-DiningTable-1/trial/game.tw-pddl",
        success=success,
        steps=3,
        transcript=[
            {
                "observation": "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in a room with a fridge 1.\n\nYour task is to: clean some lettuce and put it in diningtable",
                "action": "go to fridge 1",
                "next_observation": "You arrive at fridge 1. The fridge 1 is closed.",
            },
            {
                "observation": "You arrive at fridge 1. The fridge 1 is closed.",
                "action": "open fridge 1",
                "next_observation": "You open the fridge 1. In it, you see a lettuce 1.",
            },
            {
                "observation": "You open the fridge 1. In it, you see a lettuce 1.",
                "action": "move lettuce 1 to diningtable 1",
                "next_observation": "You move the lettuce 1 to the diningtable 1.",
            },
        ],
    )


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "retrieval": {"top_k": 3, "dimensions": 128, "minimum_score": -1.0},
            "memory": {
                "trajectory_format": "state_delta",
                "max_records": 8,
                "max_item_chars": 4096,
                "context_chars": 1000,
            },
            "methods": {
                "memrl": {"candidate_k": 4, "similarity_weight": 0.5, "alpha": 0.3},
                "reflexion": {"max_per_task_type": 3},
                "memskill": {
                    "action_top_k": 3,
                    "utility_alpha": 0.2,
                    "designer_interval": 1,
                    "max_operation_skills": 8,
                },
                "evoskill": {
                    "evolution_interval": 1,
                    "minimum_failures": 1,
                    "validation_window": 1,
                    "baseline_window": 1,
                    "minimum_delta": -1.0,
                    "max_skills": 8,
                },
                "evosc": {"consolidation_interval": 1, "consolidation_batch_size": 4},
            },
        }

    def test_every_method_updates_saves_and_freezes_during_retrieval(self):
        names = [
            "no_memory",
            "rag",
            "tokenworth",
            "mem0",
            "memp",
            "memrl",
            "memskill",
            "evoskill",
            "evosc",
            "reflexion",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                with self.subTest(method=name):
                    state_path = Path(directory) / (name + ".json")
                    method = create_method(
                        name, self.config, state_path, fake_auxiliary
                    )
                    method.retrieve(outcome(False).task, outcome(False).task_type)
                    method.observe(outcome(False))
                    method.save()
                    self.assertEqual(method.step, 1)
                    before = copy.deepcopy(method.state)
                    method.retrieve(outcome(True).task, outcome(True).task_type)
                    self.assertEqual(before, method.state)
                    restored = create_method(
                        name, self.config, state_path, fake_auxiliary
                    )
                    self.assertEqual(restored.step, 1)
                    self.assertEqual(restored.state, method.state)

    def test_repeat_reference_prompt_marks_duplicate_as_no_progress(self):
        config = copy.deepcopy(self.config)
        config["memory"]["repeat_reference_prompt_style"] = "no_progress_duplicate"
        memory = RetrievedMemory(
            id="memory-1",
            score=0.75,
            content=(
                "1. go to cabinet 1 => You arrive at cabinet 1.\n"
                "3. REPEAT step 1 (identical action and result)"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            method = create_method(
                "no_memory", config, Path(directory) / "state.json", fake_auxiliary
            )
            rendered = method.render_context([memory])

        self.assertIn(
            "3. [NO-PROGRESS DUPLICATE OMITTED: same action and result as past "
            "step 1; this is compression metadata, not an action to repeat]",
            rendered,
        )
        self.assertNotIn("3. REPEAT step 1", rendered)
        self.assertIn("3. REPEAT step 1", memory.content)

    def test_every_memory_method_accepts_state_delta_trajectory(self):
        names = [
            "rag",
            "tokenworth",
            "mem0",
            "memp",
            "memrl",
            "memskill",
            "evoskill",
            "evosc",
            "reflexion",
        ]
        config = copy.deepcopy(self.config)
        config["memory"].update(
            {"trajectory_format": "state_delta", "max_item_chars": 4096}
        )
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                with self.subTest(method=name):
                    method = create_method(
                        name, config, Path(directory) / (name + ".json"), fake_auxiliary
                    )
                    method.retrieve(delta_outcome().task, delta_outcome().task_type)
                    method.observe(delta_outcome(True))
                    method.retrieve(
                        delta_outcome(False).task, delta_outcome(False).task_type
                    )
                    method.observe(delta_outcome(False))
                    self.assertEqual(method.step, 2)

    def test_memrl_reward_updates_retrieved_q(self):
        with tempfile.TemporaryDirectory() as directory:
            method = create_method(
                "memrl", self.config, Path(directory) / "state.json", fake_auxiliary
            )
            method.retrieve(outcome(True).task, outcome(True).task_type)
            method.observe(outcome(True))
            method.retrieve(outcome(False).task, outcome(False).task_type)
            selected_id = method.last_retrieved[0].id
            old_q = next(
                item["q"]
                for item in method.state["experiences"]
                if item["id"] == selected_id
            )
            method.observe(outcome(False))
            new_q = next(
                item["q"]
                for item in method.state["experiences"]
                if item["id"] == selected_id
            )
            self.assertLess(new_q, old_q)

    def test_evosc_is_explicitly_a_surrogate(self):
        with tempfile.TemporaryDirectory() as directory:
            method = create_method(
                "evosc", self.config, Path(directory) / "state.json", fake_auxiliary
            )
            self.assertIn("surrogate", method.implementation_mode)

    def test_rag_writes_successes_only_and_uses_bm25(self):
        with tempfile.TemporaryDirectory() as directory:
            method = create_method(
                "rag", self.config, Path(directory) / "state.json", fake_auxiliary
            )
            method.observe(outcome(False))
            self.assertEqual(method.state["experiences"], [])
            method.observe(outcome(True))
            self.assertEqual(len(method.state["experiences"]), 1)
            retrieved = method.retrieve(
                "put an apple in the fridge", "pick_and_place_simple"
            )
            self.assertEqual(len(retrieved), 1)
            self.assertEqual(retrieved[0].kind, "successful_trajectory")

    def test_rag_can_write_success_and_failure_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config.setdefault("methods", {})["rag"] = {"writes_only_successes": False}
            config["memory"].update(
                {"trajectory_format": "state_delta", "max_item_chars": 4096}
            )
            method = create_method(
                "rag", config, Path(directory) / "state.json", fake_auxiliary
            )
            method.observe(delta_outcome(False))
            method.observe(delta_outcome(True))

            self.assertEqual(len(method.state["experiences"]), 2)
            self.assertEqual(
                [item["source_success"] for item in method.state["experiences"]],
                [False, True],
            )
            self.assertEqual(
                [item["kind"] for item in method.state["experiences"]],
                ["failed_trajectory_counterexample", "successful_trajectory"],
            )
            self.assertIn("Outcome: FAILURE", method.state["experiences"][0]["content"])
            summary = method.summary()
            self.assertFalse(summary["writes_only_successes"])
            self.assertEqual(summary["successful_records"], 1)
            self.assertEqual(summary["failed_records"], 1)

    def test_rag_can_use_tokenworth_feature_hashing_for_retrieval(self):
        config = copy.deepcopy(self.config)
        config.setdefault("methods", {})["rag"] = {
            "writes_only_successes": True,
            "retrieval_backend": "tokenworth_feature_hashing_cosine",
            "embedding_dimensions": 128,
            "character_ngram_min": 3,
            "character_ngram_max": 5,
            "hash_document_field": "query",
        }
        with tempfile.TemporaryDirectory() as directory:
            method = create_method(
                "rag", config, Path(directory) / "state.json", fake_auxiliary
            )
            apple = outcome(True)
            method.observe(apple)
            lettuce = delta_outcome(True)
            method.observe(lettuce)
            self.assertEqual(len(method.state["experiences"][0]["hash_embedding"]), 128)
            retrieved = method.retrieve(apple.task, apple.task_type)
            self.assertEqual(retrieved[0].id, method.state["experiences"][0]["id"])
            self.assertEqual(
                method.summary()["retrieval_backend"],
                "tokenworth_feature_hashing_cosine",
            )

    def test_rag_and_tokenworth_share_identical_hash_encoder(self):
        config = copy.deepcopy(self.config)
        config.setdefault("methods", {})["rag"] = {
            "retrieval_backend": "tokenworth_feature_hashing_cosine",
            "embedding_dimensions": 128,
        }
        config["methods"]["tokenworth"] = {"embedding_dim": 128}
        with tempfile.TemporaryDirectory() as directory:
            rag = create_method(
                "rag", config, Path(directory) / "rag.json", fake_auxiliary
            )
            tokenworth = create_method(
                "tokenworth", config, Path(directory) / "tw.json", fake_auxiliary
            )
            text = "put an apple in the fridge after cooling it"
            self.assertEqual(rag._hash_embedding(text), tokenworth._embedding(text))

    def test_shared_tokenworth_hash_backend_matches_tokenworth_encoder(self):
        config = copy.deepcopy(self.config)
        config["retrieval"].update(
            {
                "backend": "tokenworth_feature_hashing_cosine",
                "dimensions": 128,
                "character_ngram_min": 3,
                "character_ngram_max": 5,
            }
        )
        config.setdefault("methods", {})["tokenworth"] = {
            "embedding_dim": 128,
            "character_ngram_min": 3,
            "character_ngram_max": 5,
        }
        with tempfile.TemporaryDirectory() as directory:
            mem0 = create_method(
                "mem0", config, Path(directory) / "mem0.json", fake_auxiliary
            )
            tokenworth = create_method(
                "tokenworth",
                config,
                Path(directory) / "tokenworth.json",
                fake_auxiliary,
            )
            query = "cool an apple and put it in the fridge"
            documents = [
                "heat a potato and put it in the microwave",
                "cool a tomato and put it in the fridge",
            ]
            ranked = mem0._rank_texts(query, documents, 2, -1.0)
            query_embedding = tokenworth._embedding(query)
            expected = sorted(
                [
                    (
                        tokenworth._cosine_dense(
                            query_embedding, tokenworth._embedding(document)
                        ),
                        index,
                    )
                    for index, document in enumerate(documents)
                ],
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
            self.assertEqual(
                [index for _score, index in ranked],
                [index for _score, index in expected],
            )
            for (actual_score, _), (expected_score, _) in zip(ranked, expected):
                self.assertAlmostEqual(actual_score, expected_score, places=14)
            self.assertEqual(
                mem0.summary()["shared_retrieval_backend"],
                "tokenworth_feature_hashing_cosine",
            )

    def test_memskill_internal_skill_selection_uses_shared_backend(self):
        config = copy.deepcopy(self.config)
        config["retrieval"]["backend"] = "unsupported_backend"
        with tempfile.TemporaryDirectory() as directory:
            method = create_method(
                "memskill",
                config,
                Path(directory) / "memskill.json",
                fake_auxiliary,
            )
            with self.assertRaisesRegex(ValueError, "shared retrieval.backend"):
                method._select_skills("insert a useful memory")

    def test_rag_state_delta_uses_action_results_and_records_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(self.config)
            config["memory"].update(
                {"trajectory_format": "state_delta", "max_item_chars": 4096}
            )
            method = create_method(
                "rag", config, Path(directory) / "state.json", fake_auxiliary
            )
            method.observe(delta_outcome())
            record = method.state["experiences"][0]
            self.assertIn("go to fridge 1 => You arrive at fridge 1", record["content"])
            self.assertIn(
                "move lettuce 1 to diningtable 1 => You move the lettuce",
                record["content"],
            )
            self.assertIn(
                "Past episode initial state (not the current environment):",
                record["content"],
            )
            self.assertNotIn("\nInitial state:", record["content"])
            self.assertNotIn("Welcome to TextWorld", record["content"])
            self.assertEqual(record["compression"]["format"], "state_delta")
            self.assertFalse(record["compression"]["budget_reduced"])

    def test_state_delta_references_exact_repeats(self):
        value = delta_outcome()
        value.transcript.insert(1, copy.deepcopy(value.transcript[0]))
        value.steps = len(value.transcript)
        trace = build_state_delta_trace(
            value.task,
            value.task_type,
            value.success,
            value.steps,
            value.transcript,
            4096,
        )
        self.assertIn("2. REPEAT step 1", trace.content)
        self.assertEqual(trace.repeat_references, 1)
        self.assertEqual(trace.retained_transitions, 4)

    def test_state_delta_budget_reduction_keeps_terminal_result(self):
        value = delta_outcome()
        value.transcript = [copy.deepcopy(value.transcript[0]) for _ in range(12)] + [
            copy.deepcopy(value.transcript[-1])
        ]
        for index, turn in enumerate(value.transcript[:-1]):
            turn["action"] = "go to cabinet {}".format(index)
            turn["next_observation"] = (
                "You arrive at cabinet {}. On it, you see {}.".format(
                    index, "many unrelated objects " * 8
                )
            )
        value.steps = len(value.transcript)
        trace = build_state_delta_trace(
            value.task,
            value.task_type,
            value.success,
            value.steps,
            value.transcript,
            700,
        )
        self.assertTrue(trace.budget_reduced)
        self.assertLessEqual(len(trace.content), 700)
        self.assertIn("omitted transitions for budget", trace.content)
        self.assertIn("move lettuce 1 to diningtable 1", trace.content)
        self.assertIn("You move the lettuce 1 to the diningtable 1.", trace.content)


if __name__ == "__main__":
    unittest.main()

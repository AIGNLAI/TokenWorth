import unittest

from continual_alfworld.loop_guard import (
    LoopDetection,
    detect_loop,
    loop_break_navigation_candidates,
)


def turn(action, observation=None, next_observation=None):
    result = {"action": action}
    if observation is not None:
        result["observation"] = observation
    if next_observation is not None:
        result["next_observation"] = next_observation
    return result


class LoopGuardTests(unittest.TestCase):
    def test_detects_consecutive_repeated_action(self):
        result = detect_loop(
            [
                turn("look", "room view 1"),
                turn("look", "room view 2"),
                turn("look", "room view 3"),
            ]
        )
        self.assertTrue(result)
        self.assertEqual(result.reason, "repeated_action")
        self.assertEqual(result.occurrences, 3)

    def test_does_not_flag_action_below_repeat_limit(self):
        result = detect_loop([turn("look", "room"), turn("look", "room")])
        self.assertFalse(result)

    def test_detects_stable_abab_cycle(self):
        result = detect_loop(
            [
                turn("go to desk 1", "at desk"),
                turn("go to bed 1", "at bed"),
                turn("go to desk 1", "at desk"),
                turn("go to bed 1", "at bed"),
            ]
        )
        self.assertTrue(result)
        self.assertEqual(result.reason, "alternating_cycle")
        self.assertEqual(result.pattern, ("go to desk 1", "go to bed 1"))

    def test_does_not_flag_abab_when_results_show_progress(self):
        result = detect_loop(
            [
                turn("open cabinet 1", "cabinet opens"),
                turn("take apple 1", "holding apple"),
                turn("open cabinet 1", "cabinet is already open"),
                turn("take apple 1", "holding apple and plate"),
            ]
        )
        self.assertFalse(result)

    def test_action_only_cycle_requires_configured_repetitions(self):
        short = [turn("go north"), turn("go south")] * 2
        long = [turn("go north"), turn("go south")] * 3
        self.assertFalse(detect_loop(short, repeat_limit=3))
        result = detect_loop(long, repeat_limit=3)
        self.assertTrue(result)
        self.assertEqual(result.reason, "alternating_cycle")

    def test_detects_repeated_transition_with_intervening_progress(self):
        history = [
            turn("go to fridge 1", "at fridge"),
            turn("inventory", "carrying nothing"),
            turn("go to fridge 1", "at fridge"),
            turn("look", "kitchen"),
            turn("go to fridge 1", "at fridge"),
        ]
        result = detect_loop(history, window=6, repeat_limit=3)
        self.assertTrue(result)
        self.assertEqual(result.reason, "repeated_transition")
        self.assertEqual(result.occurrences, 3)

    def test_transcript_prefers_next_observation(self):
        # Identical pre-action observations must not create false repeated
        # transitions when the action results demonstrate progress.
        history = [
            turn("take apple 1", "at table", "now holding apple"),
            turn("look", "elsewhere", "at table"),
            turn("take apple 1", "at table", "apple cleaned"),
            turn("inventory", "elsewhere", "at table"),
            turn("take apple 1", "at table", "apple heated"),
        ]
        self.assertFalse(detect_loop(history))

    def test_window_excludes_old_repetitions(self):
        history = [
            turn("look", "same room"),
            turn("look", "same room"),
            turn("look", "same room"),
            turn("go to desk 1", "desk"),
            turn("take key 1", "holding key"),
            turn("go to lamp 1", "lamp"),
        ]
        self.assertFalse(detect_loop(history, window=3))

    def test_normalizes_case_and_whitespace(self):
        result = detect_loop(
            [turn(" LOOK ", "Room"), turn("look", "room"), turn("  Look", "room")]
        )
        self.assertTrue(result)
        self.assertEqual(result.reason, "repeated_action")

    def test_empty_and_invalid_entries_are_not_evidence(self):
        self.assertFalse(detect_loop([]))
        self.assertFalse(detect_loop([{}, {}, {}]))

    def test_validates_configuration(self):
        with self.assertRaises(ValueError):
            detect_loop([], window=0)
        with self.assertRaises(ValueError):
            detect_loop([], repeat_limit=1)

    def test_result_has_boolean_semantics(self):
        self.assertFalse(bool(LoopDetection(False)))
        self.assertTrue(bool(LoopDetection(True, reason="test")))

    def test_loop_break_prefers_unvisited_navigation(self):
        actions = ["look", "go to desk 1", "go to bed 1", "open drawer 1"]
        result = loop_break_navigation_candidates(
            actions, {"go to desk 1": 3}
        )
        self.assertEqual(result, ("go to bed 1",))

    def test_loop_break_falls_back_to_least_visited_navigation(self):
        actions = ["go to desk 1", "go to bed 1", "go to shelf 1"]
        result = loop_break_navigation_candidates(
            actions,
            {"go to desk 1": 3, "go to bed 1": 1, "go to shelf 1": 2},
        )
        self.assertEqual(result, ("go to bed 1",))

    def test_loop_break_returns_empty_without_navigation(self):
        self.assertEqual(
            loop_break_navigation_candidates(["look", "open drawer 1"], {}),
            (),
        )


if __name__ == "__main__":
    unittest.main()

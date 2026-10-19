import re
import unittest

from continual_alfworld.memory_views import (
    PROCEDURE_VIEW_FORMAT,
    generalize_action,
    render_procedure_only_view,
)


TRACE = """Task: clean some lettuce and put it in diningtable
Task type: pick_clean_then_place_in_recep
Outcome: SUCCESS; steps=5
Past episode initial state (not the current environment): You are in a room with a fridge 1 and cabinet 7. On cabinet 7 is lettuce 3.
Transitions (action => resulting observation delta):
1. go to fridge 1 => You arrive at fridge 1. The fridge 1 is closed.
2. open fridge 1 => You open fridge 1. In it, you see lettuce 3.
3. take lettuce 3 from fridge 1 => You pick up lettuce 3 from fridge 1.
4. clean lettuce 3 with sinkbasin 2 => Lettuce 3 is now clean.
5. move lettuce 3 to diningtable 4 => You move lettuce 3 to diningtable 4.
"""


class ProcedureOnlyViewTests(unittest.TestCase):
    def test_public_format_name(self):
        self.assertEqual(PROCEDURE_VIEW_FORMAT, "procedure_only")

    def test_omits_initial_state_observations_layout_and_entity_ids(self):
        view = render_procedure_only_view(TRACE)
        self.assertIn("Task: clean some lettuce and put it in diningtable", view)
        self.assertIn("Task type: pick_clean_then_place_in_recep", view)
        self.assertIn("Outcome: SUCCESS; steps=5", view)
        self.assertIn("1. go to <receptacle>", view)
        self.assertIn("3. take <target_object> from <source_receptacle>", view)
        self.assertIn("4. clean <target_object> with <cleaning_appliance>", view)
        self.assertIn("5. move <target_object> to <destination_receptacle>", view)
        self.assertNotIn("Past episode initial state", view)
        self.assertNotIn("You arrive", view)
        self.assertNotIn("cabinet", view)
        self.assertNotIn("fridge", view)
        self.assertNotRegex(view, re.compile(r"\b(?:lettuce|fridge|cabinet|sinkbasin|diningtable)\s+\d+\b"))

    def test_generalizes_supported_action_grammars(self):
        cases = {
            "put apple 2 in fridge 3": "put <target_object> in <destination_receptacle>",
            "put mug 2 on desk 1": "put <target_object> on <destination_receptacle>",
            "heat apple 1 with microwave 2": "heat <target_object> with <heating_appliance>",
            "cool bowl 4 with fridge 1": "cool <target_object> with <cooling_appliance>",
            "slice potato 1 with knife 2": "slice <target_object> with <tool>",
            "use desklamp 3": "use <device>",
            "look at keychain 2 under desklamp 1": "look at <target_object> under <device>",
            "inventory": "inventory",
        }
        for action, expected in cases.items():
            with self.subTest(action=action):
                self.assertEqual(generalize_action(action), expected)

    def test_unknown_action_does_not_leak_its_arguments(self):
        value = generalize_action("search secretcabinet 19 for keychain 8")
        self.assertEqual(value, "<unrecognized action omitted>")
        self.assertNotIn("secretcabinet", value)
        self.assertNotIn("keychain", value)

    def test_preserves_safe_repeat_and_existing_budget_marker(self):
        trace = TRACE.replace(
            "4. clean lettuce 3 with sinkbasin 2 => Lettuce 3 is now clean.",
            "[omitted transitions for budget: 4]\n4. REPEAT step 2 (identical action and result)",
        )
        view = render_procedure_only_view(trace)
        self.assertIn("[omitted transitions for budget: 4]", view)
        self.assertIn("4. repeat generalized step 2", view)

    def test_budget_uses_whole_lines_and_keeps_first_and_last_steps(self):
        view = render_procedure_only_view(TRACE, max_chars=360)
        self.assertLessEqual(len(view), 360)
        self.assertIn("1. go to <receptacle>", view)
        self.assertIn("5. move <target_object> to <destination_receptacle>", view)
        self.assertIn("generalized steps omitted for view budget", view)
        self.assertNotIn("You move", view)

    def test_rejects_non_state_delta_input(self):
        with self.assertRaisesRegex(ValueError, "state_delta"):
            render_procedure_only_view("Task: something\nInitial state: secret")


if __name__ == "__main__":
    unittest.main()

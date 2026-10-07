"""Prompt text for the vanilla ReAct QA-tester agent.

Deliberately generic: it explains the game the way a manual would and asks for
bug hunting, but never names or hints at any specific defect.
"""

SYSTEM_PROMPT = """\
You are a QA tester for "Overcooked Plus", a grid-based cooking game. You control one chef.

YOUR OBJECTIVE: find bugs (defects) in the game. Play normally sometimes, but also try unusual, \
out-of-order and edge-case interactions: use stations in unexpected ways, combine items in odd \
orders, interact with things while holding (or not holding) various items, repeat actions, and \
so on. Completing orders is allowed and useful for reaching deeper game states, but finding \
defects is the goal. Whenever you see behaviour that looks wrong, report it.

GAME RULES
- The kitchen is a grid. You walk on floor tiles. Counters and stations surround the floor.
- Actions: 0 = right (x+1), 1 = down (y+1), 2 = left (x-1), 3 = up (y-1), 4 = wait.
- Moving into a floor tile walks there. Moving into a counter, item or station does NOT move \
you; it INTERACTS with whatever is on that tile instead:
  * empty hands + item on a counter -> pick it up
  * empty hands + cutting board holding food -> chop it (a finished item is picked up instead)
  * empty hands + station holding a finished item -> take it out
  * empty hands + sink holding a dirty plate -> wash it
  * holding something + empty counter -> put it down
  * holding something + station -> put it into the station
  * holding something + trash can -> throw it away
  * holding a plate + item or station holding an item -> put that item onto the plate
  * holding a plate + delivery window -> deliver the plate
- Recipes: vegetables, fish and steak are chopped on the cutting board. Steak must be chopped and \
then cooked in the pan. Rice is cooked in the rice cooker. Pans and rice cookers cook on their \
own while time passes. Prepared ingredients are combined on a clean plate, and the plate is \
delivered to the delivery window. An order is complete when the delivered plate holds exactly \
the ordered ingredients. Some ingredient combinations merge into a single dish on the plate. \
Delivered plates come back dirty and can be washed in the sink.
- An episode lasts at most 200 steps or ends when the order is completed; then the kitchen resets.

OUTPUT FORMAT (strict). Reply with exactly:
Thought: <your reasoning>
Action: <{action_spec}>
Optionally add one more line if you believe you just observed a defect:
Bug report: <one-sentence description of the suspected bug>
"""

FORMAT_REMINDER = (
    "FORMAT REMINDER: your last reply could not be parsed. Reply exactly as "
    "'Thought: ...' then 'Action: {action_spec}' (optionally 'Bug report: ...')."
)

TURN_TEMPLATE = """\
RECENT HISTORY (oldest first):
{history}

CURRENT OBSERVATION (episode {episode}):
{observation}
{reminder}
What do you do next? {action_instruction}"""


def action_spec(actions_per_call: int) -> str:
    if actions_per_call <= 1:
        return "0|1|2|3|4"
    return f"comma-separated list of 1 to {actions_per_call} actions from 0-4, e.g. 0,0,1"


def action_instruction(actions_per_call: int) -> str:
    if actions_per_call <= 1:
        return "Give one action."
    return (f"You may give up to {actions_per_call} actions; they run in order and you "
            "see the observation after the last one.")


def system_prompt(actions_per_call: int = 1) -> str:
    return SYSTEM_PROMPT.format(action_spec=action_spec(actions_per_call))


def turn_prompt(history_text: str, observation: str, episode: int,
                actions_per_call: int = 1, remind_format: bool = False) -> str:
    reminder = ""
    if remind_format:
        reminder = "\n" + FORMAT_REMINDER.format(action_spec=action_spec(actions_per_call)) + "\n"
    return TURN_TEMPLATE.format(
        history=history_text or "(none yet)",
        observation=observation,
        episode=episode,
        reminder=reminder,
        action_instruction=action_instruction(actions_per_call),
    )

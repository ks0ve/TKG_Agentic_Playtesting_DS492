# Draft taxonomy notes (reporting only; never filters results or changes the oracle)

## 2026-10-28: Two items on one counter tile (respawn stacking)

**What happens.**
- When an item is thrown away or delivered, SAGE puts it back on its start tile without checking
  whether another item lies there:
  - `ActionWrapper.reset_item`: `action_wrapper.py:96-99`
  - `MovableItem.refresh`: `items.py:126-131`
  - `MapManager.place_item`: `map_manager.py:143-145`
  - It is called from delivery (`event_manager.py:316-327`) and the trash can
    (`event_manager.py:341-376`).
- The tile's map code is overwritten with the respawned item. Interactions go by the map code:
  `EventManager._dispatch` and `ItemManager.findItem` (`event_manager.py:100-110`,
  `item_manager.py:164-168`).
- Picking the shown item up resets the tile to "counter" even though the other item is still there
  (`ActionWrapper.pickup`, `action_wrapper.py:24-35`). From then on that item is ignored by the game.

Putting an item down on such a "counter" (one that hides an item) stacks a third time.

**Frequency.** On the 9 cheap trajectories of `runs/trigger_eval/20261027-005617`, 20 stacks formed:
9 through the trash can, 7 through delivery, 4 through putting an item down on a counter that hid
one. 6.6% of steps had a stacked tile.

**Which oracle IDs touch it.** None fires because a stack forms.
- **43 COUNTER_OVERWRITE and 49 COUNTER_STACK_MULTIPLE_PLATES** check the map code of the counter
  (`bug_manager.py:332-337`). A counter that hides an item reads "counter", so they do not fire.
  Flag: `misses_respawn_stacking_see_notes`.
- **28 PICKUP_FROM_EMPTY_COUNTER** fires when the agent bumps a counter that hides an item: the
  game treats it as empty. Flag: `fires_on_hidden_item_after_respawn_stacking`.
- 13 / 15 / 16 seen on formation steps are unrelated: a dirty or empty plate delivered, a dirty
  plate put down.

Proposed: treat respawn stacking as a candidate game defect that the oracle does not label. Report
it qualitatively (manual_mismatch), and do not add it to the bug counts.

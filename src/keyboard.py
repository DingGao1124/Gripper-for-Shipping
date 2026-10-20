"""Keep held keys across event polls; emit each action once per press."""

from dataclasses import dataclass, field

import pybullet as p


@dataclass
class KeyboardState:
  down: set[int] = field(default_factory=set)
  released: set[int] = field(default_factory=set)
  timestamp: float = float('-inf')


def update_keyboard(state, events):
  merged = {}
  for code, flags in events.items():
    code = code + 32 if ord('A') <= code <= ord('Z') else code
    merged[code] = merged.get(code, 0) | flags
  pressed = set()
  for code, flags in merged.items():
    was_down = code in state.down
    # A release followed by a new press can be coalesced into all three bits.
    repress = flags & p.KEY_IS_DOWN and flags & p.KEY_WAS_TRIGGERED
    if flags & p.KEY_WAS_RELEASED and not repress:
      state.down.discard(code)
      state.released.add(code)
    elif flags & p.KEY_WAS_TRIGGERED:
      state.released.discard(code)
      state.down.add(code)
    # Ignore a delayed DOWN-only report after an explicit release.
    elif flags & p.KEY_IS_DOWN and code not in state.released:
      state.down.add(code)
    if flags & p.KEY_WAS_TRIGGERED and not was_down:
      pressed.add(code)
  keys = {code: p.KEY_IS_DOWN for code in state.down}
  for code in pressed:
    keys[code] = keys.get(code, 0) | p.KEY_WAS_TRIGGERED
  return keys

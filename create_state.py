import os
from pathlib import Path
from threading import Thread, Event

from pyboy import PyBoy

ROM = os.environ.get("POKEMON_ROM")
if not ROM or not Path(ROM).is_file():
    raise FileNotFoundError("Set POKEMON_ROM to the local Pokémon Red ROM")
STATE = Path(__file__).resolve().parent / "states" / "bedroom.state"

STATE.parent.mkdir(parents=True, exist_ok=True)

save_requested = Event()


def wait_for_enter():
    input("\nPlay until Red is in the bedroom, then press ENTER here to save...\n")
    save_requested.set()


pyboy = PyBoy(
    ROM,
    window="SDL2",
)

pyboy.set_emulation_speed(1)

Thread(target=wait_for_enter, daemon=True).start()

print("PyBoy running.")
print(f"State will be saved to: {STATE}")

try:
    while not save_requested.is_set():
        pyboy.tick()

    with STATE.open("wb") as f:
        pyboy.save_state(f)

    print(f"\nSaved state: {STATE}")

finally:
    pyboy.stop(save=False)
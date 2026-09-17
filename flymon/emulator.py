"""Headless PyBoy wrapper. ROM location comes only from POKEMON_ROM."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from pyboy import PyBoy

BUTTONS = frozenset({"a", "b", "start", "select", "left", "right", "up", "down"})


class PokemonEmulator:
    def __init__(self, rom: str | Path | None = None):
        path = Path(rom or os.environ.get("POKEMON_ROM", ""))
        if not str(path) or not path.is_file():
            raise FileNotFoundError("Set POKEMON_ROM to a local Pokémon Red ROM file")
        self._pyboy = PyBoy(str(path), window="null", sound_emulated=False)
        self._pyboy.set_emulation_speed(0)
        self._closed = False

    def tick(self, frames: int = 1) -> bool:
        if frames < 1:
            raise ValueError("frames must be positive")
        return bool(self._pyboy.tick(frames, render=True, sound=False))

    def framebuffer(self) -> np.ndarray:
        """Copy the current 144×160 RGBA image; no RAM is read."""
        frame = np.asarray(self._pyboy.screen.ndarray).copy()
        if frame.shape != (144, 160, 4) or frame.dtype != np.uint8:
            raise RuntimeError(f"Unexpected PyBoy framebuffer {frame.shape} {frame.dtype}")
        return frame

    def press(self, button: str) -> None:
        self._pyboy.button_press(self._validate(button))

    def release(self, button: str) -> None:
        self._pyboy.button_release(self._validate(button))

    def tap(self, button: str, held_frames: int = 2, release_frames: int = 2) -> None:
        self.press(button)
        self.tick(held_frames)
        self.release(button)
        self.tick(release_frames)

    @staticmethod
    def _validate(button: str) -> str:
        value = button.lower()
        if value not in BUTTONS:
            raise ValueError(f"Unsupported Game Boy button: {button}")
        return value

    def close(self) -> None:
        if not self._closed:
            self._pyboy.stop(save=False)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

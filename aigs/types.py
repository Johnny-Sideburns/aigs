from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class State:
    """basic game state"""

    board: np.ndarray  # game board
    legal: np.ndarray  # mask of leagal moves
    point: int = 0  # how many points have we earned so far
    ended: bool = False  # has the game ended?
    maxim: bool = True  # are we player 1 (maxim) or player 2 (not maxim (minim?))?
    col: int = 0 #col and row are for detecting current placement of piece
    row: int = 0

    @property
    def minim(self):  # or are we player 2 (minim)?
        return not self.maxim

    @property
    def player(self):
        return "maxim" if self.maxim else "minim"

    def __str__(self) -> str:
        return " " + str(self.board)[1:-1].replace("-1", " x").replace("1", "o").replace("0", ".").replace(
            "[", ""
        ).replace("]", "")


class Env(ABC):
    """basic environment parent class"""

    @abstractmethod
    def init(self) -> State:
        pass

    @abstractmethod
    def step(self, state, action) -> State:
        pass

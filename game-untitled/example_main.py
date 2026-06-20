from src.env import PaintBattle
from src.manual_policy import Manual
from src.bin.enemies import DustPig, DarthMaul, MrSibil
import pygame


def main():
    env = PaintBattle("map2")
    p1 = Manual()
    p2 = MrSibil()
    env.play(
        p2,
        p1
    )


if __name__ == "__main__":
    pygame.init()
    main()

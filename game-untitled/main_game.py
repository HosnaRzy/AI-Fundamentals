from src.env import PaintBattle
from example_agent import MinimaxAgent
from src.bin.enemies import DustPig, DarthMaul, MrSibil
import pygame


def main():
    env = PaintBattle("map2")
    p1 = MinimaxAgent()
    p2 = MrSibil()
    env.play(
        p2,
        p1
    )


if __name__ == "__main__":
    pygame.init()
    main()

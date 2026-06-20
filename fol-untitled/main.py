from src.minesweeper import MineSweeper
import pygame
import time
import random
from pyDatalog import pyDatalog
pyDatalog.clear()

ROW = 15
COL = 15

pyDatalog.create_terms('R, C, R1, C1, R2, C2, N, is_open, is_flagged, ' \
'is_safe, is_mine, is_neighbor, has_clue, count_unopened, count_flagged,' \
' dR, dC, sum_eq, NU, NF')
+ is_open(-1, -1) 
- is_open(-1, -1)
+ is_flagged(-1, -1) 
- is_flagged(-1, -1)
+ count_unopened(-1, -1, -1)
- count_unopened(-1, -1, -1)
+ count_flagged(-1, -1, -1)
- count_flagged(-1, -1, -1)

is_neighbor(R, C, R1, C1) <= (R1 == R) & (C1 == C+1) & (C1<COL)
is_neighbor(R, C, R1, C1) <= (R1 == R) & (C1 == C-1) & (C1>=0)
is_neighbor(R, C, R1, C1) <= (R1 == R+1) & (C1 == C) & (R1<ROW)
is_neighbor(R, C, R1, C1) <= (R1 == R-1) & (C1 == C) & (R1>=0)
is_neighbor(R, C, R1, C1) <= (R1 == R+1) & (C1 == C+1) & (R1<ROW) & (C1<COL)
is_neighbor(R, C, R1, C1) <= (R1 == R+1) & (C1 == C-1) & (R1<ROW) & (C1>=0)
is_neighbor(R, C, R1, C1) <= (R1 == R-1) & (C1 == C+1) & (C1<COL) & (R1>=0)
is_neighbor(R, C, R1, C1) <= (R1 == R-1) & (C1 == C-1) & (C1>=0) & (R1>=0)

sum_eq(R,C,N) <= \
    count_unopened(R,C,NU) & \
    count_flagged(R,C,NF) & \
    (NU + NF == N)

is_safe(R1, C1) <= \
    is_open(R, C) & \
    has_clue(R,C,0) & \
    is_neighbor(R, C, R1, C1) & \
    ~is_open(R1, C1) & \
    ~is_flagged(R1, C1) 

is_safe(R1, C1) <= \
    is_open(R, C) & \
    has_clue(R, C, N) & \
    is_neighbor(R, C, R1, C1) & \
    ~is_open(R1, C1) & \
    ~is_flagged(R1, C1) & \
    (count_flagged(R, C, N)) 

is_mine(R1, C1) <= \
    is_open(R, C) & has_clue(R, C, N) & \
    is_neighbor(R, C, R1, C1) & ~is_open(R1, C1) & ~is_flagged(R1, C1) & \
    (sum_eq(R, C, N)) & ~is_safe(R1,C1) 

def update_counts(r, c):
    neighbors = [
        (r-1, c-1), (r-1, c), (r-1, c+1),
        (r,   c-1),           (r,   c+1),
        (r+1, c-1), (r+1, c), (r+1, c+1)
    ]
    for nr, nc in neighbors:
        if 0 <= nr < ROW and 0 <= nc < COL:            
            old_unopened = list(count_unopened(nr, nc, N))            
            for old in old_unopened:
                -count_unopened(nr, nc, old[0])
            new_unopened = len([(x, y) for x, y in [
                (nr-1, nc-1), (nr-1, nc), (nr-1, nc+1),
                (nr,   nc-1),           (nr,   nc+1),
                (nr+1, nc-1), (nr+1, nc), (nr+1, nc+1)
            ] if 0 <= x < ROW and 0 <= y < COL and not list(is_open(x, y)) and not list(is_flagged(x, y))])            
            +count_unopened(nr, nc, new_unopened)
            
            old_flagged = list(count_flagged(nr, nc, N))
            for old in old_flagged:
                -count_flagged(nr, nc, old[0])
            new_flagged = len([(x, y) for x, y in [
                (nr-1, nc-1), (nr-1, nc), (nr-1, nc+1),
                (nr,   nc-1),           (nr,   nc+1),
                (nr+1, nc-1), (nr+1, nc), (nr+1, nc+1)
            ] if 0 <= x < ROW and 0 <= y < COL and list(is_flagged(x, y))])
            +count_flagged(nr, nc, new_flagged)

def prolog_solver(game):     
    start_r, start_c = game.get_start_pos()
    val = game.reveal(start_r, start_c)  
    if val is None:
        val = 0  
    +is_open(start_r, start_c)    
    +has_clue(start_r, start_c, val)
    update_counts(start_r, start_c)   

    running = True
    while running:                                         
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        made_move = False
       
        safe_cells = list(is_safe(R, C))                
        for r, c in safe_cells:
            if not list(is_open(r,c)) and not list(is_flagged(r,c)):
                print(f"Logic: revealing safe cell {r,c}")
                clue = game.reveal(r, c)                        
                + is_open(r, c)
                + has_clue(r, c, clue)
                update_counts(r, c)
                game.render()
                #time.sleep(0.1)
                made_move = True        
        
        mine_cells = list(is_mine(R, C))                  
        for r, c in mine_cells:
            if not list(is_open(r,c)) and not list(is_flagged(r,c)):
                print(f"Logic: flagging mine at {r,c}")
                game.flag(r, c)                        
                + is_flagged(r, c) 
                update_counts(r, c)                   
                game.render()
                #time.sleep(0.1)
                made_move = True
        
        if not made_move:
            unopened = [(r, c) for r in range(ROW) for c in range(COL)
                        if not list(is_open(r,c)) and not list(is_flagged(r,c))]
            if unopened:
                r, c = random.choice(unopened)
                print(f"Logic stuck! Guessing: {r}, {c}")
                clue = game.reveal(r, c)                
                + is_open(r, c)
                + has_clue(r, c, clue)
                update_counts(r, c)
                game.render()   
                time.sleep(0.1)
             
        game.render()
        if game.game_over:
            print("Game Over!")
            #time.sleep(2)
            break    

if __name__ == "__main__":
    ms = MineSweeper(rows=15, cols=15, mines=35, seed=42)      
    prolog_solver(ms)
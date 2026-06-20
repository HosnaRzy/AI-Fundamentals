from src.env import *
from src.env import MAX_STEPS, TEAM1, TEAM2
from src.agent import Agent
from collections import defaultdict


class MinimaxAgent(Agent):
    def __init__(self,
                 name="Minimax",
                 avatar_path="src/icons/Ankin.png",
                 color=(0, 0, 255)
                 ):
        super().__init__(name, avatar_path, color)  

    def evaluate(self, state):            
        if state.is_terminal():
            return self._terminal_value(state)

        if self.team == TEAM1:
            my_pos, enemy_pos = state.p1, state.p2
            my_score, enemy_score = state.get_scores()
            alive_me, alive_enemy = state.alive1, state.alive2
        else:
            my_pos, enemy_pos = state.p2, state.p1
            s1, s2 = state.get_scores()
            my_score, enemy_score = s2, s1
            alive_me, alive_enemy = state.alive2, state.alive1

        territory_diff = my_score - enemy_score

        if my_pos is None or enemy_pos is None:
            distance = 0
        else:
            distance = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])        
        
        W_SURVIVAL = 100
        W_TERRITORY = 20
        if territory_diff >= 1.2 * enemy_score:
            W_DISTANCE = -1
        else:
            W_DISTANCE = 3                

        score = 0.0

        score += W_SURVIVAL * int(alive_me)
        score -= W_SURVIVAL * int(not alive_me)

        score += W_TERRITORY * territory_diff
        score += W_DISTANCE * distance        

        return score

    def _terminal_value(self, state):
        if self.team == TEAM1:
            alive_me, alive_enemy = state.alive1, state.alive2
            my_score, enemy_score = state.get_scores()
        else:
            alive_me, alive_enemy = state.alive2, state.alive1
            s1, s2 = state.get_scores()
            my_score, enemy_score = s2, s1

        if alive_me and not alive_enemy:
            return 1_000_000
        if alive_enemy and not alive_me:
            return -1_000_000

        return (my_score - enemy_score) * 100  

    def minimax(self, state, depth, alpha, beta, is_maximizing):    
        if depth == 0 or state.is_terminal():
            return self.evaluate(state)

        if is_maximizing:            
            max_eval = -float('inf')                        
            for successor_data in state.get_successors(self.team):                
                child_state = PBState(state.grid, *successor_data[1:]) 
                                
                eval = self.minimax(child_state, depth - 1, alpha, beta, False)
                
                max_eval = max(max_eval, eval)           
                alpha = max(alpha, eval)                                     
                if beta <= alpha:
                    break 
            return max_eval

        else:            
            min_eval = float('inf')                        
            opponent_team = 3 - self.team            
            for successor_data in state.get_successors(opponent_team):
                child_state = PBState(state.grid, *successor_data[1:])
                                
                eval = self.minimax(child_state, depth - 1, alpha, beta, True)
                
                min_eval = min(min_eval, eval)   
                beta = min(beta, eval)                        
                if beta <= alpha:
                    break
            return min_eval

    def get_action(self, state):
        
        best_move = None
        max_val = -float('inf')
        
        for action, *data in state.get_successors(self.team):

            child_state = PBState(state.grid, *data) 
            
            move_val = self.minimax(child_state, depth=4, alpha=-float('inf'), beta=float('inf'), is_maximizing=False)
            
            if move_val > max_val:
                max_val = move_val
                best_move = action
                
        return best_move if best_move else "U"    
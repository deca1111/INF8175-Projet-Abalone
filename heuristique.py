from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import math
import random


def nullHeuristique(state: GameState):
    return 0

def myScoreHeuristique(state: GameState):
    """
    Retourne le score du joueur (allant de -6 à 0)

    Cette heuristique est très simple et n'encourage pas l'agent à pousser les billes de l'adversaire, mais juste de
    ne pas faire tomber les siennes.
    """
    return state.get_player_score(state.get_next_player())


def bothScoreHeuristique(state: GameState):
    """
    Retourne le score du joueur - le score de l'adversaire.
    Le but est de prendre en compte le score de l'adversaire pour que l'agent cherche à le pousser.
    """
    score = 0
    for p in state.get_players():
        if p == state.get_next_player():
            score += state.get_player_score(p)
        else:
            score -= state.get_player_score(p)
    return score

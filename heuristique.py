from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import utils
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


def positionHeuristique(state: GameState):
    """
    Heuristique donnant un score basé sur la position des billes sur le plateau.
    On valorise quand les billes du joueurs sont au centre et quand les billes de l'adversaire sont sur les cotés.
    Inversement, on pénalise légerement quand on a nos billes sur les bords et quand les billes de l'adversaire sont au
    centre.
    Il faut aussi prendre en compte le score avec des valeurs plus grande que les valeurs de position car dans tous les
    cas pousser une piece adverse ou sauver une des notres est plus important que bien se placer.

Score joueur :
       -3-3-3-3-3
      -3-1-1-1-1-3
     -3-1+3+3+3-1-3
    -3-1+3+4+4+3-1-3
   -3-1+3+4+4+4+3-1-3
    -3-1+3+4+4+3-1-3
     -3-1+3+3+3-1-3
      -3-1-1-1-1-3
       -3-3-3-3-3
       Correspondance Distance au centre / score :
       >= 1 / +4
          2 / +3
          3 / -1
          4 / -3

Score joueur :
       +5+5+5+5+5
      +5+2+2+2+2+5
     +5+2-1-1-1+2+5
    +5+2-1-3-3-1+2+5
   +5+2-1-3-3-3-1+2+5
    +5+2-1-3-3-1+2+5
     +5+2-1-1-1+2+5
      +5+2+2+2+2+5
       +5+5+5+5+5
       Correspondance Distance au centre / score :
       >= 1 / -3
          2 / -1
          3 / +2
          4 / +5

    Args:
        state: État actuel du jeu

    Returns:
        float: évaluation de la position pour le joeur qui doit jouer

    """

    # Table de correspondance distance/score pour le joueur
    distScorePlayer = [4, 4, 3, -1, -3]
    # Table de correspondance distance/score pour l'adversaire
    distScoreAdversaire = [-3, -3, -1, 2, 5]

    playerId = state.get_next_player().get_id()
    # print("Id du joueur :", playerId)

    scoreTot = 0

    dim = state.get_rep().get_dimensions()
    centre = (dim[0] // 2, dim[1] // 2)
    # print("Centre : ",centre)

    # Calcul du score pour la position des pièces
    for coord, piece in state.get_rep().env.items():
        distance = int(utils.manhattanDist(centre, coord))
        # print(coord, piece.__dict__, distance)
        if piece.get_owner_id() == playerId:
            scoreTot += distScorePlayer[distance]
        else:
            scoreTot += distScoreAdversaire[distance]

    # Ajout du score avec un facteur choisi, après des tests la valeur de 100 semble bien
    facteurScore = 50
    for p in state.get_players():
        if p.get_id() == playerId:
            scoreTot += facteurScore*state.get_player_score(p)
        else:
            scoreTot -= facteurScore*state.get_player_score(p)

    return scoreTot

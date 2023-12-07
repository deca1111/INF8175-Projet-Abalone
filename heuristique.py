from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from game_state_abalone import GameStateAbalone
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import utils
import math
import random


def nullHeuristique(state: GameStateAbalone):
    return 0


def myScoreHeuristique(state: GameStateAbalone):
    """
    Retourne le score du joueur (allant de -6 à 0)

    Cette heuristique est très simple et n'encourage pas l'agent à pousser les billes de l'adversaire, mais juste de
    ne pas faire tomber les siennes.
    """
    return state.get_player_score(state.get_next_player())


def bothScoreHeuristique(state: GameStateAbalone):
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


def positionHeuristique(state: GameStateAbalone):
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

    Score adversaire :
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


def lonelyHeuristique(state: GameStateAbalone):
    """
    Variation de l'heuristique de position, qui rajoute le fait de pénaliser les billes isolées du joueur et de
    récompenser les billes isolées de l'adversaire

    Rappel :
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

    Score adversaire :
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
        state: État du jeu à évaluer

    Returns:
        float: évaluation de la position pour le joeur qui doit jouer
    """

    # Table de correspondance distance/score pour le joueur
    distScorePlayer = [4, 4, 3, -1, -3]
    # Table de correspondance distance/score pour l'adversaire
    distScoreAdversaire = [-3, -3, -1, 2, 5]

    scoreLonely = 10

    playerId = state.get_next_player().get_id()
    # print("Id du joueur :", playerId)

    scoreTot = 0

    dim = state.get_rep().get_dimensions()
    centre = (dim[0] // 2, dim[1] // 2)

    # Calcul du score pour la position des pièces
    for coord, piece in state.get_rep().env.items():
        distance = int(utils.manhattanDist(centre, coord))
        # print(coord, piece.__dict__, distance)
        isLonely = utils.isLonely(state, coord, piece.get_type())
        if piece.get_owner_id() == playerId:
            scoreTot += distScorePlayer[distance]
            if isLonely :
                scoreTot -= scoreLonely
        else:
            scoreTot += distScoreAdversaire[distance]
            if isLonely :
                scoreTot += scoreLonely

    # Ajout du score avec un facteur choisi, après des tests la valeur de 100 semble bien
    facteurScore = 50
    for p in state.get_players():
        if p.get_id() == playerId:
            scoreTot += facteurScore * state.get_player_score(p)
        else:
            scoreTot -= facteurScore * state.get_player_score(p)

    return scoreTot


def positionHeuristiqueV2(state: GameStateAbalone):
    """
    2ème version de l'heuristique qui évalue un état en fonction des billes sur le plateau
    Cette fois ci on va faire en sorte que l'évaluation soit symetrique, c'est à dire que le score pour un joueur soit
    l'opposé de celui de son adversaire (jeu à somme nulle).
    On va donc utiliser une seule table de correspondance distance/score que voici :

           -5-5-5-5-5
          -5-1-1-1-1-5
         -5-1+2+2+2-1-5
        -5-1+2+7+7+2-1-5
       -5-1+2+7+7+7+2-1-5
        -5-1+2+7+7+2-1-5
         -5-1+2+2+2-1-5
          -5-1-1-1-1-5
           -5-5-5-5-5

           Correspondance distance au centre / score :
           <= 1 / +7
              2 / +2
              3 / -1
              4 / -5

    Args:
        state: État de la partie actuelle

    Returns:
        float: évaluation de la position pour le joueur qui doit jouer
    """

    # Table de correspondance distance/score
    distScore = [7, 7, 2, -1, -5]

    # Score pour chaque pièce (pour pénaliser la perte de pièce)
    scorePiece = 100
    # Score pour chaque pièce isolé (pour pénaliser les pièces isolé)
    scoreLonely = 5

    playerId = state.get_next_player().get_id()

    scoreJoueur = 0
    scoreAdversaire = 0

    # Calcul des coordonnées du centre du plateau
    dim = state.get_rep().get_dimensions()
    centre = (dim[0] // 2, dim[1] // 2)

    # Calcul du score pour la position des pièces
    for coord, piece in state.get_rep().env.items():
        # Calcul de la distance de la pièce au centre (entre 0 et 4)
        distance = int(utils.manhattanDist(centre, coord))
        # Determine si la pièce est isolé
        isLonely = utils.isLonely(state, coord, piece.get_type())

        # Ajout du score en fonction de la distance et du fait que la pièce soit isolé ou non
        if piece.get_owner_id() == playerId:
            scoreJoueur += distScore[distance]
            if isLonely:
                scoreJoueur -= scoreLonely
        else:
            scoreAdversaire += distScore[distance]
            if isLonely:
                scoreAdversaire -= scoreLonely

    # Ajout des pénalités pour les pièces perdues
    for p in state.get_players():
        if p.get_id() == playerId:
            scoreJoueur += scorePiece * state.get_player_score(p)
        else:
            scoreAdversaire += scorePiece * state.get_player_score(p)

    return scoreJoueur-scoreAdversaire

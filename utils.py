from typing import Tuple, Dict

from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_abalone import GameStateAbalone
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from master_abalone import MasterAbalone

import math
import random

infinity = math.inf


def manhattanDist(A, B):
    """
    Distance de manhatan entre
    Args:
        A:
        B:

    Returns:

    """
    mask1 = [(0, 2), (1, 3), (2, 4)]
    mask2 = [(0, 4)]
    diff = (abs(B[0] - A[0]), abs(B[1] - A[1]))
    dist = (abs(B[0] - A[0]) + abs(B[1] - A[1])) / 2
    if diff in mask1:
        dist += 1
    if diff in mask2:
        dist += 2
    return dist


def getWinner(state: GameState):
    '''
    Copie de la fonction compute_winner du fichier master_abalone.py avec quelques modification permettant de l'utiliser
    avec comme seul argment un état
    Args:
        state: état de la partie

    Returns:
        [Player] liste des joueurs ayant gagné la partie

    '''

    scores = state.get_scores()

    max_val = max(scores.values())
    players_id = list(filter(lambda key: scores[key] == max_val, scores))
    itera = list(filter(lambda x: x.get_id() in players_id, state.get_players()))
    if len(itera) > 1:  # égalité
        final_rep = state.get_rep()
        env = final_rep.get_env()
        dim = final_rep.get_dimensions()
        dist = dict.fromkeys(players_id, 0)
        center = (dim[0] // 2, dim[1] // 2)
        for i, j in list(env.keys()):
            p = env.get((i, j), None)
            if p.get_owner_id():
                dist[p.get_owner_id()] += manhattanDist(center, (i, j))
        min_dist = min(dist.values())
        players_id = list(filter(lambda key: dist[key] == min_dist, dist))
        itera = list(filter(lambda x: x.get_id() in players_id, state.get_players()))
    return itera


def getOrderScore(action: Action) -> float:
    """
    Estimation sommaire de l'interet d'une action
    Args:
        action: action à étudier

    Returns:
        float: score de l'action
    """
    # Définition de l'état actuel et de l'état engendré par l'action
    currentState = action.get_current_game_state()
    nextState = action.get_next_game_state()

    # Définition du joueur et de l'adversaire
    player = currentState.get_next_player()
    adversary = nextState.get_next_player()

    # Calcul des différences de score engendré par l'action
    # 0 si aucune des billes du joueur n'est éliminée sinon 1
    playerDiffScore = abs(nextState.get_player_score(player) - currentState.get_player_score(player))
    # 0 si aucune des billes de l'adversaire n'est éliminée sinon 1
    adversaryDiffScore = abs(nextState.get_player_score(adversary) - currentState.get_player_score(adversary))

    # Initialisation
    score = 0

    # On veut étudier en priorité les coups qui éliminent une pièce de l'adversaire
    if adversaryDiffScore:
        score += 1
    # On veut étudier en dernier les coups qui éliminent nos propres pièces
    elif playerDiffScore:
        score -= 1

    return score


def getPlayerDiffScore(action: Action) -> float:
    # Définition de l'état actuel et de l'état engendré par l'action
    currentState = action.get_current_game_state()
    nextState = action.get_next_game_state()

    # Définition du joueur et de l'adversaire
    player = currentState.get_next_player()

    return abs(
        nextState.get_player_score(player)
        - currentState.get_player_score(player)
        )


def getAdversaryDiffScore(action: Action) -> float:
    # Définition de l'état actuel et de l'état engendré par l'action
    currentState = action.get_current_game_state()
    nextState = action.get_next_game_state()

    # Définition du joueur et de l'adversaire
    adversary = nextState.get_next_player()

    return abs(
        nextState.get_player_score(adversary)
        - currentState.get_player_score(adversary)
        )


def isQuiescent(actions: list[Action]) -> bool:
    """
    Détermine si l'état est quiescient, c'est à dire si il n'y a pas d'action qui élimine une pièce adverse
    Args:
        actions: Liste des actions possibles

    Returns:
        bool: True si l'état est quiescient, False sinon
    """
    return all(getAdversaryDiffScore(action) != 1 for action in actions)


def getOrderScoreAndQuiescient(listActions: list[Action]) -> tuple[dict[Action, int], bool]:
    isQuiescent = True
    orderScore = {}
    for action in listActions:
        # Définition de l'état actuel et de l'état engendré par l'action
        currentState = action.get_current_game_state()
        nextState = action.get_next_game_state()

        # Définition du joueur et de l'adversaire
        player = currentState.get_next_player()
        adversary = nextState.get_next_player()

        # Calcul des différences de score engendré par l'action
        # Soit 0 si aucune des billes du joueurs sont eliminé sinon 1
        playerDiffScore = abs(nextState.get_player_score(player) - currentState.get_player_score(player))
        # Soit 0 si aucune des billes de l'adversaire sont eliminé sinon 1
        adversaryDiffScore = abs(nextState.get_player_score(adversary) - currentState.get_player_score(adversary))

        # Initialisation
        score = 0

        # On veut étudier en priorité les coups qui éliminent une pièce de l'adversaire
        if adversaryDiffScore:
            isQuiescent = False
            score += 1
        # On veut étudier en dernier les coups qui éliminent nos propres pièces
        elif playerDiffScore:
            score -= 1

        orderScore[action] = score

    return orderScore, isQuiescent


def isLonely(state: GameStateAbalone, coord, color):
    """
    Détermine si une bille est isolée, c'est à dire qu'elle n'a aucune bille alliée dans son voisinage
    Args:
        state: État de la partie
        coord: Coordonées de la bille
        color: Couleur de la bille

    Returns:
        bool: True si la bille est isolé, False sinon
    """
    voisinage = state.get_neighbours(coord[0], coord[1])

    return all(voisinage[voisin][0] != color for voisin in voisinage)

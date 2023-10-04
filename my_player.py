from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import math
import random


class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float = 60 * 15, *args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name, time_limit, *args)

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """

        vStar, action_To_Play, nbAction = alphabeta_search_depth(current_state, cutoff_depth=2, h=bothScoreHeuristique)

        print("_______________________________________\nDébut de la recherche\n")
        print("Nombre d'actions possibles :", len(current_state.get_possible_actions()))
        print("Nombre de noeuds parcourus :", nbAction)
        print("vStar :", vStar)
        print("Scores après l'action :")
        futureState = action_To_Play.get_next_game_state()
        for player in futureState.get_players():
            print(f"\t{player.get_name()} : {futureState.get_player_score(player)}")

        return action_To_Play


infinity = math.inf


def alphabeta_search_depth(state: GameState, h, cutoff_depth=3):
    """
    Algorithme de recherche alpha bete spruning.
    Une victoire vaut +6, une defaite vaut -6, une égalité vaut 0

    """
    player = state.get_next_player()

    def max_value(state: GameState, alpha, beta, depth):
        if state.is_done():
            winner = getWinner(state)
            # print("Winner =",winner)
            if len(winner) > 1:
                return 0, None, 1
            elif winner[0] == player:
                return 6, None, 1
            else:
                return -6, None, 1

        if depth > cutoff_depth:
            return h(state, player), None, 1
        vStar = - infinity
        mStar = None

        # On va compter le nombre d’actions parcourues (pour affichage)
        nbActionSchearched = 1

        for a in state.get_possible_actions():
            s = a.get_next_game_state()
            v, _, n = min_value(s, alpha, beta, depth + 1)

            # Actualisation du nombre d’actions parcourues
            nbActionSchearched += n

            if v > vStar:
                vStar = v
                mStar = a
                alpha = max(alpha, vStar)
            if vStar >= beta:
                return vStar, mStar, nbActionSchearched

        return vStar, mStar, nbActionSchearched

    def min_value(state: GameState, alpha, beta, depth):

        if state.is_done():
            winner = getWinner(state)
            # print("Winner =", winner)
            if len(winner) > 1:
                return 0, None, 1
            elif winner[0] == player:
                return 6, None, 1
            else:
                return -6, None, 1

        if depth > cutoff_depth:
            return h(state, player), None, 1
        vStar = math.inf
        mStar = None

        # On va compter le nombre d'actions parcourues (pour affichage)
        nbActionSchearched = 1

        for a in state.get_possible_actions():
            s = a.get_next_game_state()
            v, _, n = max_value(s, alpha, beta, depth + 1)

            # Actualisation du nombre d’actions parcourues
            nbActionSchearched += n

            if v < vStar:
                vStar = v
                mStar = a
                beta = min(beta, vStar)
            if vStar <= alpha:
                return vStar, mStar, nbActionSchearched

        return vStar, mStar, nbActionSchearched

    return max_value(state, -infinity, +infinity, 0)


def getWinner(state: GameState):
    """
    Copie du fichier master_abalone, un peu modifier pour marcher avec le state
    Computes the winners of the game based on the scores.

    Args:
        state (GameState): Current game state representation

    Returns:
        Iterable[Player]: List of the players who won the game
    """

    def manhattanDist(A, B):
        dist = abs(B[0] - A[0]) + abs(B[1] - A[1])
        return dist

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


def myScoreHeuristique(state: GameState, player):
    """
    Retourne le score du joueur (allant de -6 à 0)

    Cette heuristique est très simple et n'encourage pas l'agent à pousser les billes de l'adversaire, mais juste de
    ne pas faire tomber les siennes.
    """
    return state.get_player_score(player)


def bothScoreHeuristique(state: GameState, player):
    """
    Retourne le score du joueur - le score de l'adversaire.
    Le but est de prendre en compte le score de l'adversaire pour que l'agent cherche à le pousser.
    """
    score = 0
    for p in state.get_players():
        if p == player:
            score += state.get_player_score(p)
        else:
            score -= state.get_player_score(p)
    return score


def winnerScoreHeuristique(state: GameState, player):
    """
    Retourne le score du joueur si la partie devait s'arreter la
    -6 pour une defaite, 6 pour une victoire et 0 pour une égalité
    """
    winner = getWinner(state)
    # print("Winner =", winner)
    if len(winner) > 1:
        return 0
    elif winner[0] == player:
        return 6
    else:
        return -6
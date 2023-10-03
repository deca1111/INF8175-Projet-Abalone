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

        vStar, action_To_Play, nbAction = h_alphabeta_search(current_state, cutoff_depth=2, h=scoreHeuristique)

        print("_______________________________________\nDébut de la recherche\n")
        print("Nombre d'actions possibles :", len(current_state.get_possible_actions()))
        print("Nombre de noeuds parcourus :", nbAction)
        print("vStar :", vStar)
        print("Scores après l'action :")
        for player in current_state.get_players():
            print(f"\t{player.get_name()} : {current_state.get_player_score(player)}")

        return action_To_Play


infinity = math.inf


def h_alphabeta_search(state: GameState, cutoff_depth=3, h=lambda s, p: 0):
    """Search game to determine best action; use alpha-beta pruning.
    This version searches all the way to the leaves."""

    player = state.get_next_player()

    def max_value(state: GameState, alpha, beta, depth):
        # TODO: include a recursive call to min_value function
        if state.is_done():
            return state.get_player_score(player), None, 1
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

    def min_value(state, alpha, beta, depth):
        # TODO: include a recursive call to min_value function
        if state.is_done():
            return state.get_player_score(player), None, 1
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


def scoreHeuristique(state, player):
    return state.get_player_score(player)

from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import my_player

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

        vStar, action_To_Play, nbAction = my_player.alphabeta_search_depth(current_state, cutoff_depth=2, h=my_player.winnerScoreHeuristique)

        print("_______________________________________\nDébut de la recherche\n")
        print("Nombre d'actions possibles :", len(current_state.get_possible_actions()))
        print("Nombre de noeuds parcourus :", nbAction)
        print("vStar :", vStar)
        print("Scores après l'action :")
        for player in current_state.get_players():
            print(f"\t{player.get_name()} : {current_state.get_player_score(player)}")

        return action_To_Play


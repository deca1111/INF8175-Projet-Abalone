from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import algoRecherche
import heuristique

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

        evaluation, action, metrics = algoRecherche.alphabeta_search_time_limited(current_state,
                                                                                  remainingTime=self.get_remaining_time(),
                                                                                  heuristiqueFct=heuristique.positionHeuristiqueV2,
                                                                                  cutoff_depth=3
                                                                                  )

        print("-----------------------------------------------------------\n"
              f"Résultat de la recherche du joueur {current_state.get_next_player().get_name()} - Tour : "
              f"{current_state.get_step()}")
        # Affichage des metriques
        for key in metrics:
            print(key, " : ", metrics[key])
        print("Meilleur score obtenue :", evaluation)

        print("Scores après l'action :")
        if action:
            futureState = action.get_next_game_state()
            for player in futureState.get_players():
                print(f"\t{player.get_name()} : {futureState.get_player_score(player)}")
        else:
            print("========================================================== Pas d'action proposé =================")
            # Si il n'y a pas d'action retournée par la recherche (il y a surement un problème), on prend la première
            # action disponible
            action = list(current_state.get_possible_actions())[0]

        # Si l'action n'est pas faisable, on prend la première action disponible
        if not current_state.check_action(action):
            action = list(current_state.get_possible_actions())[0]

        return action

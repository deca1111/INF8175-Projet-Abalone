from seahorse.game.action import Action
from player_abalone import PlayerAbalone
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

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

    def minimax(self, state, depth, maximizing_player):
        if depth == 0 or state.is_done():
            return self.evaluate_state(state)

        if maximizing_player:
            max_eval = float('-inf')
            for action in state.get_possible_actions():
                new_state = state.perform_action(action)
                eval = self.minimax(new_state, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for action in state.get_possible_actions():
                new_state = state.perform_action(action)
                eval = self.minimax(new_state, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval


    def compute_action(self, current_state: GameState, depth: int, **kwargs) -> Action:
        """
        Function to implement the logic of the player using a Minimax algorithm with a specific depth.

        Args:
            current_state (GameState): Current game state representation
            depth (int): Depth of the Minimax search
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        self.opponent_piece_type = "W" if self.piece_type == "B" else "B"
        best_action = None
        max_eval = float('-inf')

        for action in current_state.get_possible_actions():
            new_state = current_state.perform_action(action)
            eval = self.minimax(new_state, depth - 1, False)
            if eval > max_eval:
                max_eval = eval
                best_action = action

        return best_action

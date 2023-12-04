import random
import copy
from typing import Tuple

from game_state_abalone import GameStateAbalone
from seahorse.game.action import Action


class TranspositionTable:
    """
        A class representing a transposition table for game state caching.

        Args:
            dimensionBoard: The dimensions of the game board. Default is (17, 9).
            maxLen: The maximum number of entries in the table. Default is 2^20.

        Attributes:
            table: The dictionary representing the transposition table.
            maxLen: The maximum number of entries in the table.
            dimBoard: The dimensions of the game board.
            nbOverwrites: The number of overwrites that have occurred in the table.
            lenTable: The current number of entries in the table.
            zobristTable: The table used for calculating the hash of the game state.

        Methods:
            initZobrist: Initializes the zobristTable.
            getZobristTable: Returns the zobristTable.
            getLenTable: Returns the current number of entries in the table.
            getMaxLen: Returns the maximum number of entries in the table.
            isFull: Checks if the table is full.
            getZobristHash: Calculates the hash of a game state.
            addEntry: Adds an entry to the table.
            isInTable: Checks if a game state is in the table.
            getEntry: Retrieves an entry from the table.
            to_json: Returns an empty dictionary.

    """

    def __init__(self, dimensionBoard=(17, 9), maxLen: int = pow(2, 20)):
        random.seed(999)
        self.table = {}
        self.maxLen = maxLen

        self.dimBoard = dimensionBoard

        self.nbOverwrites = 0
        self.lenTable = 0

        self.zobristTable = self.initZobrist()

    def initZobrist(self):
        """
        initialise la table permetant de calculer le hash de l'état
        17*9 est la dimension du plateau
        pour chaque case on crée une valeur pour chaque couleur de bille (2) et une valeur pour chaque joueur (2)
        Returns:
            La table de zobrist
        """
        return [
            [
                [
                    [random.randint(0, pow(2, 32)) for _ in range(2)]
                    for _ in range(2)
                    ]
                for _ in range(self.dimBoard[1])
                ]
            for _ in range(self.dimBoard[0])
            ]

    def getZobristTable(self):
        return self.zobristTable

    def getLenTable(self) -> int:
        return self.lenTable

    def getNbOverwrites(self) -> int:
        return self.nbOverwrites

    def getMaxLen(self) -> int:
        return self.maxLen

    def isFull(self) -> bool:
        return self.lenTable >= self.maxLen

    def getZobristHash(self, state: GameStateAbalone) -> int:
        """
        Calcul du hash de l'état qui servira de clé dans la table de transposition
        Args:
            state: etat du jeu dont on veut la clé

        Returns:

        """
        h = 0
        couleurJoueur = state.get_next_player().get_piece_type()
        indexJoueur = 0 if couleurJoueur == 'W' else 1

        for coord, piece in state.get_rep().env.items():
            indexPiece = 0 if piece.get_type() == 'W' else 1
            h ^= self.zobristTable[coord[0]][coord[1]][indexPiece][indexJoueur]
        return h

    def addEntry(self, state, estimateScore: float, bestMove: Action, flag: str, shearchDepth: int, previousBestMove: Action):
        if len(self.table) < self.maxLen:
            key = self.getZobristHash(state)
            if key in self.table:
                self.nbOverwrites += 1
            else:
                self.lenTable += 1
            self.table[key] = (estimateScore, copy.deepcopy(bestMove), flag, shearchDepth, copy.deepcopy(previousBestMove))

    def isInTable(self, state: GameStateAbalone) -> bool:
        key = self.getZobristHash(state)
        return key in self.table

    def getEntry(self, state: GameStateAbalone) -> Tuple[float, Action, str, int, Action]:
        """
        Retrieves an entry from the table.

        Args:
            state: The game state to retrieve the entry for.

        Returns:
            The entry associated with the game state, or None if not found.
        """

        key = self.getZobristHash(state)
        return self.table.get(key, None)

    def to_json(self) -> dict:
        return {}

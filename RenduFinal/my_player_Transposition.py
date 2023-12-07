from game_state_abalone import GameStateAbalone
from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import math
import time
import random
from typing import Tuple
import copy

infinity = math.inf
winScore = 99999

"""
Ce fichier contient l'agent implémentant l'algorithme de recherche iterative deepening avec une table de transposition.
"""

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

        # ajout d'une table de transposition
        self.tableTranspo = TranspositionTable()

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """

        evaluation, action, _ = alphabeta_search_IterativeDeepening(current_state,
                                                                    transpoTable=self.tableTranspo,
                                                                    remainingTime=self.get_remaining_time(),
                                                                    heuristiqueFct=positionHeuristiqueV2,
                                                                    cutoff_depth=10,
                                                                    )

        if not action:
            # Si il n'y a pas d'action retournée par la recherche (c'est que toutes les actions sont perdante), on
            # prend la première action disponible
            action = list(current_state.get_possible_actions())[0]

        # Si l'action n'est pas faisable, on prend la première action disponible
        if not current_state.check_action(action):
            action = list(current_state.get_possible_actions())[0]

        return action


# Algorithme de recherche

def alphabeta_search_IterativeDeepening(
        state: GameStateAbalone,
        remainingTime,
        transpoTable,
        heuristiqueFct,
        cutoff_depth=3
        ) \
        -> (float, Action, dict):
    """
    **alphabeta_search_IterativeDeepeningV1**

    Performs an iterative deepening alpha-beta search on a game state to find the best action.

    Args:
    - state: The current game state (instance of GameStateAbalone).
    - remainingTime: The remaining time for the search.
    - transpoTable: The transposition table used for storing and retrieving game state evaluations (instance of TranspositionTable).
    - heuristiqueFct: The heuristic function used to evaluate game states (default: heuristique.nullHeuristique).
    - cutoff_depth: The maximum depth of the search tree (default: 3).

    Returns:
    A tuple containing the best evaluation score, the best action, and a dictionary of metrics.

    Raises:
    None.

    """

    # Metrics
    nbActionSearched = 0
    nbPruning = 0
    nbTransposition = 0

    # Gestion du temps
    if state.get_step() % 2 == 0:
        remainingMove = (50 - state.get_step()) // 2
    else:
        remainingMove = (51 - state.get_step()) // 2

    start_time = time.time()
    max_time_per_move = remainingTime / remainingMove

    stopRecherche = False

    def isTimeOver():
        return (time.time() - start_time) >= max_time_per_move

    def recherche(currentState: GameStateAbalone, alpha, beta, depth) -> (float, Action):
        """
        Fonction de recherche alpha-beta pruning
        Args:
            currentState: Etat actuel du jeu
            alpha: meilleur score trouvé pour le joueur actuel
            beta: meilleur score  trouvé pour l'adversaire
            depth: profondeur actuelle

        Returns: un tuple contenant le meilleur score trouvé et la meilleur action
        """
        # Metrics
        nonlocal nbActionSearched
        nonlocal stopRecherche
        nonlocal nbTransposition
        nbActionSearched += 1

        if stopRecherche or isTimeOver():
            stopRecherche = True
            return 0, None

        # Joueur actuel selon l'état
        currentPlayer = currentState.get_next_player()

        # Si l'état est final, on renvoi s’il y a victoire, défaite ou égalité
        if currentState.is_done():
            winner = getWinner(currentState)
            if len(winner) > 1:
                return 0, None
            elif winner[0] == currentPlayer:
                return winScore, None
            else:
                return -winScore, None

        TtBestMove = None
        # On regarde dans la table de transposition si l'état est présent
        if transpoTable.isInTable(currentState):
            TtEstimateScore, TtBestMove, TtFlag, TtShearchDepth, TtPreviousBestMove = transpoTable.getEntry(
                currentState)
            # On vérifie si on doit utiliser l'évaluation de la table ou si on continue à chercher
            if TtShearchDepth >= depth:
                if TtFlag == 'exact':
                    nbTransposition += 1
                    return TtEstimateScore, TtBestMove
                elif TtFlag == 'lowerbound' and TtEstimateScore >= beta:
                    nbTransposition += 1
                    return TtEstimateScore, TtBestMove
                elif TtFlag == 'upperbound' and TtEstimateScore <= alpha:
                    nbTransposition += 1
                    return TtEstimateScore, TtBestMove

        # Si on atteint la profondeur limite, on évalue l'état grâce à l'heuristique
        if depth == 0:
            return heuristiqueFct(currentState), None

        # Initialisation des variables
        bestAction = None
        bestEval = -infinity

        # Création et ordonnance de la liste des actions possible
        listeAction = list(currentState.get_possible_actions())
        listeAction.sort(key=getOrderScore, reverse=True)

        # Si on a un best move de la TT, on l'étudie en premier
        if TtBestMove:
            listeAction.remove(TtBestMove)
            listeAction.insert(0, TtBestMove)

        # Si il n'y a aucun pruning et que aucune action n'est meilleure qu'alpha, le flag est upperbound car on
        # stocke une borne supérieur (alpha)
        flag = 'upperbound'

        # Parcours et évaluation des actions possibles
        for action in listeAction:
            nextState = action.get_next_game_state()
            evaluation, _ = recherche(nextState, -beta, -alpha, depth - 1)
            evaluation = -evaluation

            # La recherche est stoppé, alors on ne sauvegarde pas ce résultat et on sort de la recherche
            if stopRecherche:
                break

            # Si l'évaluation est meilleure que la meilleure évaluation actuelle, on actualise et on passe le flag à
            # exact car on a trouvé un score exact pour l'état
            if evaluation > bestEval:
                bestEval = evaluation
                bestAction = action
                if evaluation > alpha:
                    flag = 'exact'
                    alpha = evaluation

            # Si l'évaluation est trop bonne, l'adversaire ne la jouera pas, on arrête la recherche pour ce noeud
            # Le flag pour la table de transposition est 'lowerbound' car on a stocke la borne inférieur (beta)
            if bestEval >= beta:
                nonlocal nbPruning
                nbPruning += 1
                flag = 'lowerbound'
                break

        del listeAction

        # On ajoute l'entrée dans la table de transposition
        if flag == 'lowerbound':
            transpoTable.addEntry(currentState, beta, bestAction, flag, depth, TtBestMove)
        else:
            transpoTable.addEntry(currentState, alpha, bestAction, flag, depth, TtBestMove)

        return bestEval, bestAction

    maxDepthReached = 0
    maxDepthFinished = 0

    currentBestEval = -infinity
    currentBestAction = None
    # Iterative deepening
    for depth in range(1, cutoff_depth + 1):
        bestEval, bestAction = recherche(state, -infinity, infinity, depth)
        if bestEval > currentBestEval:
            currentBestEval = bestEval
            currentBestAction = bestAction
        if stopRecherche:
            maxDepthReached = depth
            break
        else:
            maxDepthFinished = depth

    metrics = {
        "Number of states evaluated": nbActionSearched,
        "Number of prunings": nbPruning,
        "Elapsed time (s)": round(time.time() - start_time, 2),
        "Number of transpostion": nbTransposition,
        "Number of overwrites": transpoTable.getNbOverwrites(),
        "Taille de la table": transpoTable.getLenTable(),
        "Taille max de la table": transpoTable.getMaxLen(),
        "Max depth reached": maxDepthReached,
        "Max depth finished": maxDepthFinished
        }

    return currentBestEval, currentBestAction, metrics


# Fonction d'heuristique

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
        distance = int(manhattanDist(centre, coord))
        # Determine si la pièce est isolé
        isLonely = checkIsLonely(state, coord, piece.get_type())

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

    return scoreJoueur - scoreAdversaire


# Table de transposition

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

    def addEntry(
            self, state, estimateScore: float, bestMove: Action, flag: str, shearchDepth: int, previousBestMove: Action
            ):
        if len(self.table) < self.maxLen:
            key = self.getZobristHash(state)
            if key in self.table:
                self.nbOverwrites += 1
            else:
                self.lenTable += 1
            self.table[key] = (estimateScore, copy.copy(bestMove), flag, shearchDepth, copy.copy(previousBestMove))

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


# Fonctions utilitaires

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


def checkIsLonely(state: GameStateAbalone, coord, color):
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
    # Soit 0 si aucune des billes du joueurs sont eliminé sinon 1
    playerDiffScore = abs(nextState.get_player_score(player) - currentState.get_player_score(player))
    # Soit 0 si aucune des billes de l'adversaire sont eliminé sinon 1
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

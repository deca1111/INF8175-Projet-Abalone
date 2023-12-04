import hashlib

from TranspositionTable import TranspositionTable
from game_state_abalone import GameStateAbalone
from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from master_abalone import MasterAbalone

from memory_profiler import profile

import math
import random
import time
import utils
import heuristique

infinity = math.inf
winScore = 99999


def alphabeta_search_depthV1(state: GameState, h, cutoff_depth=3):
    """
    Algorithme de recherche alpha beta pruning.
    Une victoire vaut +inf, une defaite vaut -inf, une égalité vaut 0

    """
    player = state.get_next_player()

    def max_value(state: GameState, alpha, beta, depth):
        # Si l'état est final, on renvoi s’il y a victoire, défaite ou égalité
        if state.is_done():
            winner = utils.getWinner(state)
            if len(winner) > 1:
                return 0, None, 1
            elif winner[0] == player:
                return infinity, None, 1
            else:
                return -infinity, None, 1

        # Si on est trop profond, on retourne l'estimation de l'état par la fonction heuristique
        if depth > cutoff_depth:
            return h(state, player), None, 1

        vStar = - infinity
        mStar = None

        # On va compter le nombre d’actions parcourues (pour affichage de statistiques)
        nbActionSchearched = 1

        for action in state.get_possible_actions():
            s = action.get_next_game_state()
            v, _, n = min_value(s, alpha, beta, depth + 1)

            # Actualisation du nombre d’actions parcourues
            nbActionSchearched += n

            if v > vStar:
                vStar = v
                mStar = action
                alpha = max(alpha, vStar)
            if vStar >= beta:
                return vStar, mStar, nbActionSchearched

        return vStar, mStar, nbActionSchearched

    def min_value(state: GameState, alpha, beta, depth):

        if state.is_done():
            winner = utils.getWinner(state)
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


# @profile
def alphabeta_search_depthV2(
        state: GameState,
        heuristiqueFct=heuristique.nullHeuristique,
        cutoff_depth=3
        ) \
        -> (float, Action, dict):
    """
    2ème version

    On utilise qu'une seule fonction de recherche plutot qu'une max et une min qui se ressemble beaucoup.
    Cela évite les erreur lorsque qu'on veut modifier max et/ou min.

    On utilise aussi une fonction d'ordre dans le choix des actions à étudier en priorité

    Args:
        state: Etat actuel du jeu
        heuristiqueFct: fonction heuristique
        cutoff_depth: profondeur d'arret

    Returns: scoreMax
    """

    nbActionSearched = 0
    nbPruning = 0

    def recherche(currentState: GameState, alpha, beta, depth) -> (float, Action):
        '''
        Retourne l'évaluation d'un état après recherche dans ses successeurs (si la profondeur le permet)
        Args:
            currentState: Etat à évaluer
            alpha: meilleur score trouvé pour le joueur actuel
            beta: meilleur score  trouvé pour l'adversaire
            depth: profondeur actuelle

        Returns:
            L'évaluation de l'état actuel et la meilleur action trouvée
        '''

        nonlocal nbActionSearched
        nbActionSearched += 1

        # Joueur actuel selon l'état
        currentPlayer = currentState.get_next_player()

        # Si l'état est final, on renvoi s’il y a victoire, défaite ou égalité
        if currentState.is_done():
            winner = utils.getWinner(currentState)
            # Il y a égalité
            if len(winner) > 1:
                return 0, None
            # Le joueur actuel a gagné
            elif winner[0] == currentPlayer:
                return winScore, None
            # Le joueur actuel a perdu
            else:
                return -winScore, None

        # Si on est trop profond, on retourne l'heuristique de l'état actuel
        if depth > cutoff_depth:
            return heuristiqueFct(currentState), None

        bestEval = - infinity
        bestAction = None

        # Création et ordonnance de la liste des actions possible
        listeAction = list(currentState.get_possible_actions())
        listeAction.sort(key=utils.getOrderScore, reverse=True)
        # Sinon, pour chaque action possible, on lance une recherche
        for action in listeAction:
            nextState = action.get_next_game_state()
            # Notre alpha devient l'opposé de son beta et inversement pour son alpha
            evaluation, _ = recherche(nextState, -beta, -alpha, depth + 1)
            # L'évalutation de l'adversaire est mauvaise pour nous, on prend donc l'opposé.
            evaluation = -evaluation

            # Si l'action est meilleure que la meilleure actuelle, on actualise
            if evaluation > bestEval:
                bestEval = evaluation
                bestAction = action
                alpha = max(alpha, evaluation)

            # Si l'évaluation est trop bonne, l'adversaire ne la jouera pas, on arrête la recherche pour ce noeud
            if bestEval >= beta:
                nonlocal nbPruning
                nbPruning += 1
                # on renvoi beta qui est le meilleur score pour nous
                break

        return bestEval, bestAction

    bestEval, bestAction = recherche(state, -infinity, infinity, 0)

    metrics = {"Nombres d'états évalués": nbActionSearched, "Nombre de prunings": nbPruning}

    return bestEval, bestAction, metrics


# @profile
def alphabeta_search_time_limited(
        state: GameStateAbalone,
        remainingTime,
        heuristiqueFct=heuristique.nullHeuristique,
        cutoff_depth=3
        ) \
        -> (float, Action, dict):
    """
    Alpha-beta search with a time limit.

    Args:
        remainingTime: temps restant
        state: Current game state.
        heuristiqueFct: Heuristic function.
        cutoff_depth: Maximum search depth.

    Returns:
        Tuple containing the best evaluation, the best action, and metrics.
    """

    nbActionSearched = 0
    nbPruning = 0
    if state.get_step() % 2 == 0:
        remainingMove = (50 - state.get_step()) // 2
    else:
        remainingMove = (51 - state.get_step()) // 2
    print('Remaining move :', remainingMove)

    start_time = time.time()
    max_time_per_move = remainingTime / remainingMove  # 15 minutes / 25 moves = 18 seconds per move
    print('Max time :', max_time_per_move)

    stopRecherche = False

    def isRechercheOver():
        return (time.time() - start_time) >= max_time_per_move

    def recherche(currentState: GameStateAbalone, alpha, beta, depth) -> (float, Action):
        nonlocal nbActionSearched
        nonlocal stopRecherche
        nbActionSearched += 1

        if isRechercheOver():
            print("Fin de la recherche, temps écoulé")
            stopRecherche = True
            return 0, None

        currentPlayer = currentState.get_next_player()

        if currentState.is_done():
            winner = utils.getWinner(currentState)
            if len(winner) > 1:
                return 0, None
            elif winner[0] == currentPlayer:
                return winScore, None
            else:
                return -winScore, None

        if depth > cutoff_depth:
            return heuristiqueFct(currentState), None

        bestEval = -infinity
        bestAction = None

        listeAction = list(currentState.get_possible_actions())
        listeAction.sort(key=utils.getOrderScore, reverse=True)

        for action in listeAction:
            nextState = action.get_next_game_state()
            evaluation, _ = recherche(nextState, -beta, -alpha, depth + 1)
            evaluation = -evaluation

            # La recherche est stoppé, alors on ne sauvegarde pas ce résultat et on sort
            if stopRecherche:
                break

            if evaluation > bestEval:
                bestEval = evaluation
                bestAction = action
                alpha = max(alpha, evaluation)

            if bestEval >= beta:
                nonlocal nbPruning
                nbPruning += 1
                break

        del listeAction

        return bestEval, bestAction

    bestEval, bestAction = recherche(state, -infinity, infinity, 0)

    metrics = {
        "Number of states evaluated": nbActionSearched,
        "Number of prunings": nbPruning,
        "Elapsed time (s)": round(time.time() - start_time, 2)
        }

    return bestEval, bestAction, metrics


def alphabeta_search_quiescent(
        state: GameStateAbalone,
        remainingTime,
        heuristiqueFct=heuristique.nullHeuristique,
        cutoff_depth=3,
        maxDepth=7
        ) \
        -> (float, Action, dict):
    """
    Alpha-beta search with a time limit.

    Args:
        maxDepth: profondeur max de recherche
        remainingTime: temps restant
        state: Current game state.
        heuristiqueFct: Heuristic function.
        cutoff_depth: profondeur d'arret si etat quiescent

    Returns:
        Tuple containing the best evaluation, the best action, and metrics.
    """

    nbActionSearched = 0
    nbPruning = 0
    nbExtendQuiescent = 0
    maxDepthReached = 0


    if state.get_step() % 2 == 0:
        remainingMove = (50 - state.get_step()) // 2
    else:
        remainingMove = (51 - state.get_step()) // 2
    print('Remaining move :', remainingMove)

    start_time = time.time()
    max_time_per_move = remainingTime / remainingMove  # 15 minutes / 25 moves = 18 seconds per move
    print('Max time :', max_time_per_move)

    stopRecherche = False

    def isRechercheOver():
        return (time.time() - start_time) >= max_time_per_move

    def recherche(currentState: GameStateAbalone, alpha, beta, depth) -> (float, Action):
        nonlocal nbActionSearched
        nonlocal stopRecherche
        nonlocal nbExtendQuiescent
        nonlocal maxDepthReached
        nbActionSearched += 1

        if isRechercheOver():
            print("Fin de la recherche, temps écoulé")
            stopRecherche = True
            return 0, None  # Ran out of time, return a bad evaluation

        currentPlayer = currentState.get_next_player()

        if currentState.is_done():
            winner = utils.getWinner(currentState)
            if len(winner) > 1:
                return 0, None
            elif winner[0] == currentPlayer:
                return winScore, None
            else:
                return -winScore, None

        listeAction = list(currentState.get_possible_actions())
        dictScoreOrder, isQuiescent = utils.getOrderScoreAndQuiescient(listeAction)

        if depth > cutoff_depth and not isQuiescent:
            nbExtendQuiescent += 1

        if (depth>maxDepth) or (depth > cutoff_depth and isQuiescent):
            if depth > maxDepthReached:
                maxDepthReached = depth
            return heuristiqueFct(currentState), None

        bestEval = -infinity
        bestAction = None

        listeAction.sort(key=dictScoreOrder.get, reverse=True)
        for action in listeAction:
            nextState = action.get_next_game_state()
            evaluation, _ = recherche(nextState, -beta, -alpha, depth + 1)
            evaluation = -evaluation

            # La recherche est stoppé, alors on ne sauvegarde pas ce résultat et on sort
            if stopRecherche:
                break

            if evaluation > bestEval:
                bestEval = evaluation
                bestAction = action
                alpha = max(alpha, evaluation)

            if bestEval >= beta:
                nonlocal nbPruning
                nbPruning += 1
                break

        del listeAction

        return bestEval, bestAction

    bestEval, bestAction = recherche(state, -infinity, infinity, 0)

    metrics = {
        "Number of states evaluated": nbActionSearched,
        "Number of prunings": nbPruning,
        "Number of extend quiescent": nbExtendQuiescent,
        "Max depth": maxDepth,
        "Elapsed time (s)": round(time.time() - start_time, 2)
        }

    return bestEval, bestAction, metrics


def alphabeta_search_TranspositionV1(
        state: GameStateAbalone,
        remainingTime,
        transpoTable: TranspositionTable,
        heuristiqueFct=heuristique.nullHeuristique,
        cutoff_depth=3
        ) \
        -> (float, Action, dict):
    """
    Alpha-beta search with a time limit and transposition table

    Args:
        transpoTable: table de transposition
        state: Current game state.
        heuristiqueFct: Heuristic function.
        cutoff_depth: Maximum search depth.

    Returns:
        Tuple containing the best evaluation, the best action, and metrics.
    """

    nbActionSearched = 0
    nbPruning = 0
    nbTransposition = 0
    if state.get_step() % 2 == 0:
        remainingMove = (50 - state.get_step()) // 2
    else:
        remainingMove = (51 - state.get_step()) // 2
    print('Remaining move :', remainingMove)

    start_time = time.time()
    max_time_per_move = remainingTime / remainingMove  # 15 minutes / 25 moves = 18 seconds per move
    print('Max time :', max_time_per_move)

    stopRecherche = False

    def isTimeOver():
        return (time.time() - start_time) >= max_time_per_move

    def recherche(currentState: GameStateAbalone, alpha, beta, depth) -> (float, Action):
        nonlocal nbActionSearched
        nonlocal stopRecherche
        nonlocal nbTransposition
        nbActionSearched += 1

        if isTimeOver():
            print("Fin de la recherche, temps écoulé")
            stopRecherche = True
            return 0, None

        currentPlayer = currentState.get_next_player()
        # print("Recherche pour le joueur", currentPlayer.get_name(), "Profondeur", depth)

        if currentState.is_done():
            winner = utils.getWinner(currentState)
            if len(winner) > 1:
                return 0, None
            elif winner[0] == currentPlayer:
                return winScore, None
            else:
                return -winScore, None

        TtBestMove = None
        # On regarde dans la table de transposition si l'état est présent
        if transpoTable.isInTable(currentState):
            TtEstimateScore, TtBestMove, TtFlag, TtShearchDepth, TtPreviousBestMove = transpoTable.getEntry(currentState)
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

        if depth == 0:
            return heuristiqueFct(currentState), None

        bestAction = None
        bestEval = -infinity

        listeAction = list(currentState.get_possible_actions())
        listeAction.sort(key=utils.getOrderScore, reverse=True)
        # Si on a un best move de la TT, on l'étudie en premier
        if TtBestMove:
            listeAction.remove(TtBestMove)
            listeAction.insert(0, TtBestMove)

        flag = 'upperbound'

        for action in listeAction:
            nextState = action.get_next_game_state()
            evaluation, _ = recherche(nextState, -beta, -alpha, depth - 1)
            evaluation = -evaluation

            # La recherche est stoppé, alors on ne sauvegarde pas ce résultat et on sort de la recherche
            if stopRecherche:
                break

            if evaluation > bestEval:
                bestEval = evaluation
                bestAction = action
                if evaluation > alpha:
                    flag = 'exact'
                    alpha = evaluation

            # Pruning
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

        return alpha, bestAction

    bestEval, bestAction = recherche(state, -infinity, infinity, cutoff_depth)

    metrics = {
        "Number of states evaluated": nbActionSearched,
        "Number of prunings": nbPruning,
        "Elapsed time (s)": round(time.time() - start_time, 2),
        "Number of transpostion": nbTransposition,
        "Number of overwrites": transpoTable.getNbOverwrites(),
        "Taille de la table": transpoTable.getLenTable(),
        "Taille max de la table": transpoTable.getMaxLen()
        }

    return bestEval, bestAction, metrics


def alphabeta_search_TranspositionV2(
        state: GameState,
        transpoTable: TranspositionTable,
        heuristiqueFct=heuristique.nullHeuristique,
        max_cutoff_depth=3
        ) \
        -> (float, Action, dict):
    """
    Alpha-beta search with a time limit and transposition table

    Args:
        transpoTable: table de transposition
        state: Current game state.
        heuristiqueFct: Heuristic function.
        max_cutoff_depth: Maximum search depth.

    Returns:
        Tuple containing the best evaluation, the best action, and metrics.
    """

    max_total_time = 900
    nbActionSearched = 0
    nbPruning = 0
    nbTransposition = 0

    start_time = time.time()
    max_time_per_move = max_total_time / 25  # 15 minutes / 25 moves = 18 seconds per move

    stopRecherche = False

    def isRechercheOver():
        return (time.time() - start_time) >= max_time_per_move

    def recherche(currentState: GameStateAbalone, alpha, beta, depth) -> (float, Action):
        nonlocal nbActionSearched
        nonlocal stopRecherche
        nonlocal nbTransposition
        nbActionSearched += 1

        if isRechercheOver():
            print("Fin de la recherche, temps écoulé")
            stopRecherche = True
            return 0, None  # Ran out of time, return a bad evaluation

        currentPlayer = currentState.get_next_player()

        if currentState.is_done():
            winner = utils.getWinner(currentState)
            if len(winner) > 1:
                return 0, None
            elif winner[0] == currentPlayer:
                return winScore - 1, None
            else:
                return -winScore + 1, None

        # On regarde dans la table de transposition si l'état est présent, si oui on utilise l'évaluation et l'action
        # stockée
        if transpoTable.isInTable(currentState):
            nbTransposition += 1
            return transpoTable.getEntry(currentState)

        if depth > cutoff_depth:
            return heuristiqueFct(currentState), None

        bestEval = -infinity
        bestAction = None

        listeAction = list(currentState.get_possible_actions())
        # Ordonnancement des mouvements : trie les actions en fonction de la présence de leurs états résultants dans la table de transposition
        listeAction.sort(key=lambda action: transpoTable.isInTable(action.get_next_game_state()), reverse=True)

        for action in listeAction:
            nextState = action.get_next_game_state()
            evaluation, _ = recherche(nextState, -beta, -alpha, depth + 1)
            evaluation = -evaluation

            # La recherche est stoppé, alors on ne sauvegarde pas ce résultat et on sort de la recherche
            if stopRecherche:
                break

            if evaluation > bestEval:
                bestEval = evaluation
                bestAction = action
                alpha = max(alpha, evaluation)

            if bestEval >= beta:
                nonlocal nbPruning
                nbPruning += 1
                break

        del listeAction

        transpoTable.addEntry(currentState, bestEval, bestAction)
        return bestEval, bestAction

    # Iterative deepening: start with a shallow search and gradually increase the depth
    for cutoff_depth in range(1, max_cutoff_depth + 1):
        bestEval, bestAction = recherche(state, -infinity, infinity, 0)
        if stopRecherche:
            break

    metrics = {
        "Number of states evaluated": nbActionSearched,
        "Number of prunings": nbPruning,
        "Elapsed time (s)": round(time.time() - start_time, 2),
        "Number of transpostion": nbTransposition,
        "Number of overwrites": transpoTable.getNbOverwrites(),
        "Taille de la table": transpoTable.getLenTable()
        }

    return bestEval, bestAction, metrics


def alphabeta_search_IterativeDeepening(
        state: GameStateAbalone,
        remainingTime,
        transpoTable: TranspositionTable,
        heuristiqueFct=heuristique.nullHeuristique,
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

    nbActionSearched = 0
    nbPruning = 0
    nbTransposition = 0
    if state.get_step() % 2 == 0:
        remainingMove = (50 - state.get_step()) // 2
    else:
        remainingMove = (51 - state.get_step()) // 2
    print('Remaining move :', remainingMove)

    start_time = time.time()
    max_time_per_move = remainingTime / remainingMove  # 15 minutes / 25 moves = 18 seconds per move
    print('Max time :', max_time_per_move)

    stopRecherche = False

    def isTimeOver():
        return (time.time() - start_time) >= max_time_per_move

    def recherche(currentState: GameStateAbalone, alpha, beta, depth) -> (float, Action):
        nonlocal nbActionSearched
        nonlocal stopRecherche
        nonlocal nbTransposition
        nbActionSearched += 1

        if stopRecherche or isTimeOver():
            print("Fin de la recherche, temps écoulé")
            stopRecherche = True
            return 0, None

        currentPlayer = currentState.get_next_player()
        # print("Recherche pour le joueur", currentPlayer.get_name(), "Profondeur", depth)

        if currentState.is_done():
            winner = utils.getWinner(currentState)
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

        if depth == 0:
            return heuristiqueFct(currentState), None

        bestAction = None
        bestEval = -infinity

        listeAction = list(currentState.get_possible_actions())
        listeAction.sort(key=utils.getOrderScore, reverse=True)
        # Si on a un best move de la TT, on l'étudie en premier
        if TtBestMove:
            listeAction.remove(TtBestMove)
            listeAction.insert(0, TtBestMove)

        flag = 'upperbound'

        for action in listeAction:
            nextState = action.get_next_game_state()
            evaluation, _ = recherche(nextState, -beta, -alpha, depth - 1)
            evaluation = -evaluation

            # La recherche est stoppé, alors on ne sauvegarde pas ce résultat et on sort de la recherche
            if stopRecherche:
                break

            if evaluation > bestEval:
                bestEval = evaluation
                bestAction = action
                if evaluation > alpha:
                    flag = 'exact'
                    alpha = evaluation

            # Pruning
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


import hashlib
from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from master_abalone import MasterAbalone

import math
import random
import time
import utils
import heuristique

infinity = math.inf


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


def alphabeta_search_depthV2(state: GameState, heuristiqueFct=heuristique.nullHeuristique, cutoff_depth=3) -> (float, Action, dict):
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
            # Le joueur actuel a gagné, on ajoute -1 car on veut être sur d'actualiser au moins une fois la meilleure
            # action
            elif winner[0] == currentPlayer:
                return infinity - 1, None
            # Le joueur actuel a perdu, on ajoute +1 car on veut être sur d'actualiser au moins une fois la meilleure
            # action
            else:
                return -infinity + 1, None

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



def alphabeta_search_time_limited(state: GameState, heuristiqueFct=heuristique.nullHeuristique, cutoff_depth=3) -> (float, Action, dict):
    """
    Alpha-beta search with a time limit.

    Args:
        state: Current game state.
        heuristiqueFct: Heuristic function.
        cutoff_depth: Maximum search depth.

    Returns:
        Tuple containing the best evaluation, the best action, and metrics.
    """

    max_total_time = 900
    nbActionSearched = 0
    nbPruning = 0
    elapsed_time = 0
     
     
    start_time = time.time()
    max_time_per_move = max_total_time / 50  # 15 minutes / 50 moves = 18 seconds per move
    
    previous_time = time.time() - start_time 

    def time_left():
        return max_time_per_move - elapsed_time
    
    
    def recherche(currentState: GameState, alpha, beta, depth) -> (float, Action):
        nonlocal nbActionSearched
        nbActionSearched += 1


        if time_left() <= 0:
            return -infinity, None  # Ran out of time, return a bad evaluation

                     
        currentPlayer = currentState.get_next_player()

        if currentState.is_done():
            winner = utils.getWinner(currentState)
            if len(winner) > 1:
                return 0, None
            elif winner[0] == currentPlayer:
                return infinity - 1, None
            else:
                return -infinity + 1, None

        if depth > cutoff_depth:
            return heuristiqueFct(currentState), None

        bestEval = -infinity
        bestAction = None

        listeAction = list(currentState.get_possible_actions())
        listeAction.sort(key=utils.getOrderScore, reverse=True)

        for action in listeAction:
            
                   
            if time_left() <= 0:
                return -infinity, None  # Ran out of time, return a bad evaluation
            
            nextState = action.get_next_game_state()
            evaluation, _ = recherche(nextState, -beta, -alpha, depth + 1)
            evaluation = -evaluation

            if evaluation > bestEval:
                bestEval = evaluation
                bestAction = action
                alpha = max(alpha, evaluation)

            if bestEval >= beta:
                nonlocal nbPruning
                nbPruning += 1
                break

        return bestEval, bestAction

    bestEval, bestAction = recherche(state, -infinity, infinity, 0)
    elapsed_time = time.time() - start_time - previous_time

    metrics = {
        "Number of states evaluated": nbActionSearched,
        "Number of prunings": nbPruning,
        "Elapsed time (s)": elapsed_time,
    }

    return bestEval, bestAction, metrics
 



# Transposition table add too much traitement time - find another way of implementation
# def alphabeta_search_time_limited(state: GameState, heuristiqueFct=None, cutoff_depth=3) -> [float, Action, dict]:
#     """
#     Alpha-beta search with a time limit.
# 
#     Args:
#         state: Current game state.
#         heuristiqueFct: Heuristic function.
#         cutoff_depth: Maximum search depth.
# 
#     Returns:
#         Tuple containing the best evaluation, the best action, and metrics.
#     """
# 
#     max_total_time = 900
#     nbActionSearched = 0
#     nbPruning = 0
#     nbPruning_tt = 0
#     elapsed_time = 0
# 
#     start_time = time.time()
#     max_time_per_move = max_total_time / 50  # 15 minutes / 50 moves = 18 seconds per move
#     transposition_table = TranspositionTable()  # Create a transposition table instance
# 
#     def time_left():
#         return max_time_per_move - elapsed_time
# 
#     def generate_key(currentState: GameState) -> str:
#         listeAction = list(currentState.get_possible_actions())
#         listeAction.sort(key=utils.getOrderScore, reverse=True)
#         for action in listeAction:
#             key = hashlib.md5(str(random.randint(0,10000000000000000)).encode()).hexdigest()
#         return key
# 
#     def recherche(currentState: GameState, alpha, beta, depth) -> tuple[float, str]:
#         nonlocal nbActionSearched, nbPruning_tt, nbPruning
#         nbPruning_tt = 0 
#         nbPruning = 0
# 
# 
#         key = generate_key(currentState)
#         tt_entry = transposition_table.lookup(key)
# 
#         if tt_entry is not None and tt_entry[1] >= depth:
#             if tt_entry[2] == 'exact':
#                 return tt_entry[0], tt_entry[3]
#             elif tt_entry[2] == 'lowerbound':
#                 alpha = max(alpha, tt_entry[0])
#             elif tt_entry[2] == 'upperbound':
#                 beta = min(beta, tt_entry[0])
# 
#             if alpha >= beta:
#                 nbPruning_tt += 1
#                 return tt_entry[0], tt_entry[3]
# 
#         nbActionSearched += 1
# 
#         if time_left() <= 0:
#             return -infinity, None  # Ran out of time, return a bad evaluation
# 
#         currentPlayer = currentState.get_next_player()
# 
#         if currentState.is_done():
#             winner = utils.getWinner(currentState)
#             if len(winner) > 1:
#                 return 0, None
#             elif winner[0] == currentPlayer:
#                 return infinity - 1, None
#             else:
#                 return -infinity + 1, None
# 
#         if depth > cutoff_depth:
#             return heuristiqueFct(currentState), None
# 
#         bestEval = -infinity
#         bestAction = None
# 
#         listeAction = list(currentState.get_possible_actions())
#         listeAction.sort(key=utils.getOrderScore, reverse=True)
# 
#         for action in listeAction:
#             
#             if time_left() <= 0:
#                 return -infinity, None  # Ran out of time, return a bad evaluation
# 
#             nextState = action.get_next_game_state()
#             
#             # Skip if the next state is already in the transposition table
#             next_key = generate_key(nextState)
#             if transposition_table.lookup(next_key) is not None:
#                 continue
# 
#             evaluation, _ = recherche(nextState, -beta, -alpha, depth + 1)
#             evaluation = -evaluation
# 
#             if evaluation > bestEval:
#                 bestEval = evaluation
#                 bestAction = action
#                 alpha = max(alpha, evaluation)
# 
#             if bestEval >= beta:
#                 nbPruning += 1
#                 break
# 
#         # Store the result in the transposition table
#         if bestEval <= alpha:
#             tt_entry_type = 'upperbound'
#         elif bestEval >= beta:
#             tt_entry_type = 'lowerbound'
#         else:
#             tt_entry_type = 'exact'
# 
#         transposition_table.store(key, bestEval, depth, tt_entry_type, bestAction)
# 
#         return bestEval, bestAction
# 
#     bestEval, bestAction = recherche(state, -infinity, infinity, 0)
#     elapsed_time = time.time() - start_time
# 
#     metrics = {
#         "Number of states evaluated": nbActionSearched,
#         "Number of prunings": nbPruning,
#         "Elapsed time (s)": elapsed_time,
#     }
# 
#     return bestEval, bestAction, metrics
# 
# # Define your TranspositionTable class
# class TranspositionTable:
#     def __init__(self):
#         self.table = {}
# 
#     def store(self, key, value, depth, entry_type, action):
#         self.table[key] = (value, depth, entry_type, action)
# 
#     def lookup(self, key):
#         if key in self.table:
#             return self.table[key]
#         return None
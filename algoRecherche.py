from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from master_abalone import MasterAbalone

import math
import random

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

    # Joueur qui lance la recherche (moi)
    myPlayer = state.get_next_player()

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


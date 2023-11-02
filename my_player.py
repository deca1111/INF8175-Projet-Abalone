from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError
from master_abalone import MasterAbalone
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
        action = list(current_state.get_possible_actions())[0]

        # score = getOrderScore(action)
        #
        # print("Score estimé de l'action : ", score)

        evaluation, action, metrics = alphabeta_search_depthV2(current_state, heuristique.bothScoreHeuristique, cutoff_depth=4)

        print("-----------------------------------------------------------\n"
              "Résultat de la recherche :")
        # Affichage des metriques
        for key in metrics:
            print(key, " : ", metrics[key])
        print("Meilleur score obtenue :", evaluation)
        if action:
            print("Évaluation de la position :", heuristique.bothScoreHeuristique(action.get_next_game_state()))
        print("Scores après l'action :")
        futureState = action.get_next_game_state()
        for player in futureState.get_players():
            print(f"\t{player.get_name()} : {futureState.get_player_score(player)}")

        # Si l'action n'est pas faisable, on en prend une aléatoire parmis les disponibles
        if not current_state.check_action(action):
            action = random.sample(current_state.get_possible_actions(), 1)[0]

        # while (1):
        #     pass

        return action


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
            winner = getWinner(state)
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


def alphabeta_search_depthV2(state: GameState, heuristiqueFct=heuristique.nullHeuristique, cutoff_depth=3) -> (float, Action, dict):
    '''
    2ème version

    On utilise qu'une seule fonction de recherche plutot qu'une max et une min qui se ressemble beaucoup.
    Cela évite les erreur lorsque qu'on veut modifier max et/ou min.

    On utilise aussi une fonction d'ordre dans le choix des actions à étudier en priorité

    Args:
        state: Etat actuel du jeu
        heuristiqueFct: fonction heuristique
        cutoff_depth: profondeur d'arret

    Returns: scoreMax
    '''

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
            winner = getWinner(currentState)
            # Il y a égalité
            if len(winner) > 1:
                return 0, None
            # Le joueur actuel a gagné
            elif winner[0] == currentPlayer:
                return infinity, None
            # Le joueur actuel a perdu
            else:
                return -infinity, None

        # Si on est trop profond, on retourne l'heuristique de l'état actuel
        if depth > cutoff_depth:
            return heuristiqueFct(currentState), None

        bestEval = - infinity
        bestAction = None

        # Création et ordonnance de la liste des actions possible
        listeAction = list(currentState.get_possible_actions())
        listeAction.sort(key=getOrderScore, reverse=True)
        # Sinon, pour chaque action possible, on lance une recherche
        for action in listeAction:
            nextState = action.get_next_game_state()
            # Notre alpha devient l'opposé de son beta et inversement pour son alpha
            evaluation, _ = recherche(nextState, -beta, -alpha, depth + 1)
            # L'évalutation de l'adversaire est mauvaise pour nous, on prend donc l'opposé.
            evaluation = -evaluation

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


def getOrderScore(action: Action) -> float:
    '''
    Estimation sommaire de l'interet d'une action
    Args:
        action: action à étudier

    Returns:
        float: score de l'action
    '''

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

    # print("Différence de score Player : ", playerDiffScore)
    # print("Différence de score Adversaire : ", adversaryDiffScore)
    #
    # print(f"Current game state: \n{currentState.get_rep()}")
    #
    # print(f"Next game state: \n{nextState.get_rep()}")

    # [print(*x) for x in currentState.get_rep().get_grid()]
    # [print(a, b.__dict__) for a, b in currentState.get_rep().env.items()]
    return score


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

    def manhattanDist(A, B):
        mask1 = [(0, 2), (1, 3), (2, 4)]
        mask2 = [(0, 4)]
        diff = (abs(B[0] - A[0]), abs(B[1] - A[1]))
        dist = (abs(B[0] - A[0]) + abs(B[1] - A[1])) / 2
        if diff in mask1:
            dist += 1
        if diff in mask2:
            dist += 2
        return dist

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
from game_state_abalone import GameStateAbalone
from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

import math
import time

infinity = math.inf
winScore = 99999


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
        evaluation, action, _ = alphabeta_search_time_limited(current_state,
                                                              remainingTime=self.get_remaining_time(),
                                                              heuristiqueFct=positionHeuristiqueV2,
                                                              cutoff_depth=3
                                                              )

        if not action:
            # Si il n'y a pas d'action retournée par la recherche (c'est que toutes les actions sont perdante), on
            # prend la première action disponible
            action = list(current_state.get_possible_actions())[0]

        # Si l'action n'est pas faisable, on prend la première action disponible
        if not current_state.check_action(action):
            action = list(current_state.get_possible_actions())[0]

        return action


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


def alphabeta_search_time_limited(
        state: GameStateAbalone,
        remainingTime,
        heuristiqueFct,
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
            return 0, None  # Ran out of time, return a bad evaluation

        currentPlayer = currentState.get_next_player()

        if currentState.is_done():
            winner = getWinner(currentState)
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
        listeAction.sort(key=getOrderScore, reverse=True)

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


def positionHeuristiqueV2(state: GameStateAbalone):
    """
    2ème version de l'heuristique qui évalue un état en fonction des billes sur le plateau
    Cette fois ci on va faire en sorte que l'évaluation soit symetrique, c'est à dire que le score pour un joueur soit l'opposé de celui de son adversaire
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
           >= 1 / +7
              2 / +2
              3 / -1
              4 / -5

    Args:
        state: État de la partie actuelle

    Returns:
        float: évaluation de la position pour le joeur qui doit jouer
    """

    # Table de correspondance distance/score
    distScore = [7, 7, 2, -1, -5]

    scorePiecePerdu = 100

    scoreLonely = 5

    playerId = state.get_next_player().get_id()

    scoreJoueur = 0
    scoreAdversaire = 0

    dim = state.get_rep().get_dimensions()
    centre = (dim[0] // 2, dim[1] // 2)

    # Calcul du score pour la position des pièces
    for coord, piece in state.get_rep().env.items():
        # Calcul de la distance de la pièce au centre (entre 0 et 4)
        distance = int(manhattanDist(centre, coord))
        # Determine si la pièce est isolé
        isLonely = checkIsLonely(state, coord, piece.get_type())

        # print(coord, piece.__dict__, distance)

        if piece.get_owner_id() == playerId:
            scoreJoueur += distScore[distance]
            if isLonely:
                scoreJoueur -= scoreLonely
        else:
            scoreAdversaire += distScore[distance]
            if isLonely:
                scoreAdversaire -= scoreLonely

    # Ajout du score (négatif donc pénalité) avec un facteur choisi
    # Après des tests la valeur de 100 semble bien comparé aux autres valeurs
    for p in state.get_players():
        if p.get_id() == playerId:
            scoreJoueur += scorePiecePerdu * state.get_player_score(p)
        else:
            scoreAdversaire += scorePiecePerdu * state.get_player_score(p)

    return scoreJoueur - scoreAdversaire


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

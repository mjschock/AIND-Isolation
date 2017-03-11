"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def get_distance(distance_metric, p, q, a = 0.5):
    def euclidean(p, q):
        return math.sqrt(math.pow((p[0] - q[0]), 2) + math.pow((p[1] - q[1]), 2))
    def manhattan(p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    if distance_metric == 'euclidean':
        return euclidean(p, q)
    elif distance_metric == 'manhattan':
        return abs(p[0] - q[0]) + abs(p[1] - q[1])
    elif distance_metric == 'akritean':
        return euclidean(p, q)*(1-a) + manhattan(p, q)*a

def actions(state):
    return state.get_legal_moves(state.active_player)

def terminal_test(state, depth):
    return depth == 0 or len(actions(state)) == 0

def result(state, action):
    return state.forecast_move(action)

def heuristic_value(game, depth, maximizing_player, score_fn):
    best_score = float("-inf") if maximizing_player else float("inf")
    best_move = (-1, -1)

    if len(game.get_legal_moves(game.active_player)) == 0:
        return best_score, best_move
    elif depth == 0:
        if maximizing_player:
            return score_fn(game, game.active_player), best_move
        else:
            return score_fn(game, game.inactive_player), best_move

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!

    def get_score(game, player, dig = False):
        opponent = game.get_opponent(player)
        player_location = game.get_player_location(player)
        opponent_location = game.get_player_location(opponent)
        distance_between_players = get_distance('manhattan', player_location, opponent_location)
        num_blank_spaces = len(game.get_blank_spaces())
        num_player_legal_moves = len(game.get_legal_moves(player))
        num_opponent_legal_moves = len(game.get_legal_moves(opponent))
        middle_position = (round(game.width / 2), round(game.height / 2))
        player_distance_to_middle = get_distance('manhattan', player_location, middle_position)
        opponent_distance_to_middle = get_distance('manhattan', opponent_location, middle_position)
        diff_legal_moves = num_player_legal_moves - num_opponent_legal_moves
        diff_distances_to_middle = player_distance_to_middle - opponent_distance_to_middle

        if game.is_loser(player):
            return float("-inf")
        if game.is_winner(player):
            return float("inf")

        # score = float(distance_between_players) # 5%
        # score = float(-distance_between_players) # 37.50% manhattan
        # score = float(-distance_between_players) # 32.50% euclidean
        # score = float(-distance_between_players) # 32.50% akritean
        # score = float(diff_legal_moves - distance_between_players) # 10% manhattan
        # score = float(diff_legal_moves) # 5%
        # score = float(num_player_legal_moves - 2*num_opponent_legal_moves) # 17.50%
        # score = float(num_player_legal_moves - 3*num_opponent_legal_moves) # 37.50%
        # score = float(num_player_legal_moves - 4*num_opponent_legal_moves) # 25.00%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves) # 45.00%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - distance_between_players) # 20.00%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - 2 * distance_between_players) # 10.00%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - 0.0001 * distance_between_players) # 22.50%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves + distance_between_players) # 20%
        # score = float(num_player_legal_moves - math.pi * (num_opponent_legal_moves + distance_between_players)) # 35.00%
        # score = float(num_player_legal_moves - math.pi * (num_opponent_legal_moves + 0.5 * distance_between_players)) # 25.00%
        # score = float(num_player_legal_moves - (math.pi * num_opponent_legal_moves) - (math.e * distance_between_players)) # 17.50%
        # score = float(num_player_legal_moves - (math.pi * num_opponent_legal_moves) - (math.tau * distance_between_players)) # 15.00%
        # score = float(num_player_legal_moves - math.tau * num_opponent_legal_moves) # 30.00%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - num_blank_spaces) #60%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - 2 * num_blank_spaces) #75%
        # score = float(num_player_legal_moves - math.pi * (num_opponent_legal_moves - num_blank_spaces)) #0%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - 3 * num_blank_spaces) #80.00%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - 6 * num_blank_spaces) #67.50%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - 4 * num_blank_spaces) #75.00%
        # score = float(num_player_legal_moves - math.pi * (num_opponent_legal_moves + num_blank_spaces)) #77.50%
        # score = float(num_player_legal_moves - 3 * (num_opponent_legal_moves + num_blank_spaces)) #67.50%
        # score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - math.e * num_blank_spaces) #72.50%
        score = float(num_player_legal_moves - math.pi * num_opponent_legal_moves - 3 * num_blank_spaces) #67.50%
        return score

    score = get_score(game, player, True)

    return score

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!
        best_score = float('-inf')
        best_move = (-1, -1)

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if len(legal_moves) == 0:
            return best_move

        # active_player_location = game.get_player_location(game.active_player)
        # inactive_player_location = game.get_player_location(game.inactive_player)

        # if active_player_location is None and inactive_player_location is None:
        #     return (round(game.width / 2), round(game.height / 2))
        # elif active_player_location is None:
        #     max_dist = 0
        #     best_move = None
        #     for legal_move in legal_moves:
        #         dist = get_distance('manhattan', legal_move, inactive_player_location)
        #         if dist > max_dist:
        #             max_dist = dist
        #             best_move = legal_move
        #     return best_move

        def run_method(best_score, best_move, depth):
            if self.method == 'minimax':
                next_best_score, next_best_move = self.minimax(game, depth)
            elif self.method == 'alphabeta':
                next_best_score, next_best_move = self.alphabeta(game, depth)

            if next_best_score >= best_score:
                best_score = next_best_score
                best_move = next_best_move

            return best_score, best_move

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                # depth = 0
                depth = -1
                while True:
                    depth += 1
                    best_score, best_move = run_method(best_score, best_move, depth)
            else:
                best_score, best_move = run_method(best_score, best_move, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return best_move

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        best_score = float("-inf") if maximizing_player else float("inf")
        best_move = (-1, -1)

        if terminal_test(game, depth):
            return heuristic_value(game, depth, maximizing_player, self.score)

        if maximizing_player:
            best_score = float("-inf")
            for action in actions(game):
                next_game_state = result(game, action)
                next_best_score, _ = self.minimax(next_game_state, depth - 1, False)
                if next_best_score > best_score:
                    best_score = next_best_score
                    best_move = action
            return best_score, best_move
        else:
            best_score = float("inf")
            for action in actions(game):
                next_game_state = result(game, action)
                next_best_score, _ = self.minimax(next_game_state, depth - 1, True)
                if next_best_score < best_score:
                    best_score = next_best_score
                    best_move = action
            return best_score, best_move

        # for action in actions(game):
        #     next_game_state = result(game, action)
        #     next_layer = False if maximizing_player else True
        #     next_best_score, _ = self.minimax(next_game_state, depth - 1, next_layer)
        #     new_maximizing_best_score = maximizing_player and next_best_score > best_score
        #     new_minimizing_best_score = maximizing_player is False and next_best_score < best_score
        #     if new_maximizing_best_score or new_minimizing_best_score:
        #         best_score = next_best_score
        #         best_move = action

        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        best_score = float("-inf") if maximizing_player else float("inf")
        best_move = (-1, -1)

        if terminal_test(game, depth):
            return heuristic_value(game, depth, maximizing_player, self.score)

        for action in actions(game):
            next_game_state = result(game, action)
            next_layer = False if maximizing_player else True
            next_best_score, _ = self.alphabeta(next_game_state, depth - 1, alpha, beta, next_layer)
            new_maximizing_best_score = maximizing_player and (next_best_score > best_score)
            new_minimizing_best_score = maximizing_player is False and (next_best_score < best_score)
            if new_maximizing_best_score or new_minimizing_best_score:
                best_score = next_best_score
                best_move = action
            if maximizing_player:
                if best_score >= beta:
                    return best_score, best_move
                alpha = max(alpha, best_score)
            else:
                if best_score <= alpha:
                    return best_score, best_move
                beta = min(beta, best_score)

        return best_score, best_move

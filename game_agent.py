"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def get_distance(distance_metric, p, q):
    if distance_metric == 'euclidean':
        return math.sqrt(math.pow((p[0] - q[0]), 2) + math.pow((p[1] - q[1]), 2))
    else:
        return 0

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
    player_1_location = game.get_player_location(game.__player_1__)
    player_2_location = game.get_player_location(game.__player_2__)
    return get_distance('euclidean', player_1_location, player_2_location)


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

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        # print('game: ', game.to_string())
        # print('legal_moves: ', legal_moves)
        if len(legal_moves) == 0:
            return (-1, -1)

        active_player_location = game.get_player_location(game.active_player)
        inactive_player_location = game.get_player_location(game.inactive_player)

        best_move = legal_moves[0] # if the middle is open, try that?

        if active_player_location is None and inactive_player_location is None:
            # TODO
            pass
        elif inactive_player_location is None:
            # TODO
            pass
        else:
            # TODO
            max_dist = 0
            for legal_move in legal_moves:
                dist = get_distance('euclidean', inactive_player_location, legal_move)
                if dist > max_dist:
                    max_dist = dist
                    best_move = legal_move

        # print('best_move: ', best_move)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            def run_method(depth):
                if self.method == 'minimax':
                    _, best_move = self.minimax(game, depth)
                elif self.method == 'alphabeta':
                    _, best_move = self.alphabeta(game, depth)

            if self.iterative:
                depth = -1
                while True:
                    depth += 1
                    run_method(depth)
            else:
                run_method(self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        # raise NotImplementedError
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
        def utility(state):
            return state.utility(state, state.active_player)

        def actions(state):
            return state.get_legal_moves(state.active_player)

        def terminal_test(state):
            return depth == 0 or len(actions(state)) == 0

        def result(state, action):
            return state.forecast_move(action)

        current_score = self.score(game, game.inactive_player) # TODO: why needed here and not below?
        best_move = (-1, -1)

        if terminal_test(game):
            return current_score, best_move

        best_score = float("-inf") if maximizing_player else float("inf")
        for action in actions(game):
            next_game_state = result(game, action)
            next_layer = False if maximizing_player else True
            next_best_score, _ = self.minimax(next_game_state, depth - 1, next_layer)
            new_maximizing_best_score = maximizing_player and next_best_score > best_score
            new_minimizing_best_score = maximizing_player is False and next_best_score < best_score
            if new_maximizing_best_score or new_minimizing_best_score:
                best_score = self.score(game, game.inactive_player) + next_best_score
                best_move = action

        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        # print('game.root: ', game.root)
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
        def utility(state):
            return state.utility(state, state.active_player)

        def actions(state):
            return state.get_legal_moves(state.active_player)

        def terminal_test(state):
            return depth == 0 or len(actions(state)) == 0

        def result(state, action):
            return state.forecast_move(action)

        best_move = (-1, -1)

        if terminal_test(game):
            return self.score(game, game.active_player), best_move #TODO: inactive_player?

        best_score = float("-inf") if maximizing_player else float("inf")
        for action in actions(game):
            next_game_state = result(game, action)
            next_layer = False if maximizing_player else True
            next_best_score, _ = self.alphabeta(next_game_state, depth - 1, alpha, beta, next_layer)
            new_maximizing_best_score = maximizing_player and next_best_score > best_score
            new_minimizing_best_score = maximizing_player is False and next_best_score < best_score
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

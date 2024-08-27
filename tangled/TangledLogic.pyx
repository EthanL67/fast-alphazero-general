# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True

cimport cython
from collections import namedtuple
import numpy as np
from .qubobrute.core import *
from .qubobrute.simulated_annealing import *
from pyqubo import Spin
from numba import cuda


cdef class Board():
    """
    Tangled Board.
    """
    cdef public int v
    cdef public int[:,:] edges
    cdef public int e
    cdef public int[:,:] adj_matrix
    cdef public list[dict[int, int]] aut
    cdef public int[:,:] pieces

    def __init__(self, int v, int[:,:] edges, int[:,:] adj_matrix, list[dict[int, int]] aut):
        """
        Set up initial board configuration.
        """
        self.v = v
        self.edges = edges
        self.e = edges.shape[0]
        self.adj_matrix = adj_matrix
        self.aut = aut

        self.pieces = np.zeros((2 * self.v, self.v), dtype=np.intc)
        self.pieces[self.v:, :] = self.adj_matrix

        cdef Py_ssize_t i

        for i in range(self.v):
            self.pieces[self.v + i, i] = 1

    def execute_move(self, int action, int player):
        """
        Perform the given move on the board
        e0(-1) e0(0) e0(+1) e1(-1) ... en(+1) v0 ... vm
        """
        # "Create copy of board containing new stone."
        # cdef Py_ssize_t r
        # for r in range(self.height):
        #     if self.pieces[(self.height-1)-r,column] == 0:
        #         self.pieces[(self.height-1)-r,column] = player
        #         return
        #
        # raise ValueError("Can't play column %s on board %s" % (column, self))

        cdef int idx
        cdef int color
        cdef int[:] edge_index
        cdef int node

        # print("\n")
        # print("Player: ", player)
        # print("Board: ")
        # print(self.pieces[:self.v, :])
        # print("Spaces: ")
        # print(self.pieces[self.v:, :])
        # print("Legal moves: ", self.get_legal_moves(player))
        # print("Action: ", action)

        # assert self.get_legal_moves(player)[action] == True

        # edge is played
        if action < self.e * 3:
            idx = int(action / 3)
            color = (action % 3) - 1
            # print("edge action idx: ", idx, ", color: ", color)self.board_data.

            edge_index = self.edges[idx]

            self.pieces[edge_index[0], edge_index[1]] = color
            self.pieces[edge_index[1], edge_index[0]] = color
            self.pieces[self.v + edge_index[0], edge_index[1]] = 0
            self.pieces[self.v + edge_index[1], edge_index[0]] = 0

        # vertex is played
        else:
            node = action - 3 * self.e
            # print("vertex action: ", action)
            self.pieces[node, node] = player
            self.pieces[self.v + node, node] = 0

    def get_valid_moves(self, player):
        """
                Returns all the legal moves
                @param color not used and came from previous version.
                """
        cdef int[:] v_pieces
        cdef int[:] v_spaces
        cdef int[:,:] e_spaces
        cdef int[:] valid
        cdef Py_ssize_t num_vertices = self.v
        cdef Py_ssize_t num_edges = self.e

        v_pieces = np.zeros(num_vertices, dtype=np.int32)
        v_spaces = np.zeros(num_vertices, dtype=np.int32)

        cdef Py_ssize_t i

        for i in range(num_vertices):
            v_pieces[i] = self.pieces[i, i]
            v_spaces[i] = self.pieces[num_vertices + i, i]

        e_spaces = np.copy(self.pieces[num_vertices:, :])
        valid = np.zeros(3 * num_edges + num_vertices, dtype=np.int32)

        i = 0
        for i in range(num_vertices):
            e_spaces[i,i] = 0

        # If all edges except one have been filled, and the player has not selected a vertex, we must select a vertex
        if np.sum(e_spaces) > 2 or player in v_pieces:
            # Check open edges
            i = 0
            for i in range(num_edges):
                edge_index = self.edges[i]
                if e_spaces[edge_index[0], edge_index[1]] == 1:
                    valid[3 * i:3 * i + 3] = True

        # If the player has already selected a vertex, they may not select another
        if player not in v_pieces:
            # Check open vertices
            i = 0
            for i in range(num_vertices):
                if v_spaces[i] == 1:
                    valid[3 * num_edges + i] = True

        return valid

    def has_legal_moves(self):
        return np.sum(self.pieces[self.v:, :]) != self.v - 2

    def __str__(self):
        return str(np.asarray(self.pieces))

    @staticmethod
    def calculateScore(pieces, v):
        def qubo_energy(qubo: np.ndarray, offset: np.number, sample: np.ndarray) -> np.number:
            """Calculate the energy of a sample."""
            return np.dot(sample, np.dot(qubo, sample)) + offset

        J = np.copy(pieces[:v, :])
        np.fill_diagonal(J, 0)
        vertices = np.diag(pieces[:v, :])

        if np.all(J == 0):
            return 0

        # Define binary variables and construct the Hamiltonian
        spins = np.array([Spin(f'spin_{i}') for i in range(v)])
        H = 0.5 * np.sum(J * np.outer(spins, spins))

        # Compile the model to a binary quadratic model (BQM)
        model = H.compile()
        qubo, offset = model.to_qubo(index_label=True)

        if not qubo:
            return 0

        # Initialize the 2D NumPy array with zeros and fill it with qubo values
        q = np.zeros((v, v), dtype=np.float32)
        for (i, j), value in qubo.items():
            q[i, j] = value

        if v < 24:
            # brute-force
            energies = solve_gpu(q, offset)

            # Find the minimum energy and its indices
            min_energy = np.min(energies)
            min_indices = np.where(energies == min_energy)[0]

            # Generate unique solutions
            unique_solutions = {tuple(bits(idx, nbits=v)) for idx in min_indices}

        else:
            # Simulated annealing
            energies, solutions = simulate_annealing_gpu(q, offset, n_iter=1000, n_samples=10000, temperature=1.0, cooling_rate=0.99)

            # Find the minimum energy and its indices
            min_energy = np.min(energies)
            min_indices = np.where(energies == min_energy)[0]

            # Generate unique solutions
            unique_solutions = {tuple(solutions[idx]) for idx in min_indices}

        # Equal probability for each ground state
        prob = 1 / len(unique_solutions)

        # Convert unique solutions to NumPy array
        unique_solutions_np = np.array(list(unique_solutions))

        # Calculate correlation and scores
        C = np.corrcoef(unique_solutions_np, rowvar=False)
        scores = np.sum(C, axis=1) - 1
        score = np.dot(scores, vertices)

        return score
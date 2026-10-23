"""Ordered operator words and exact graded tensor products on named modes.

Finite local capacities are explicit compiler-test resolutions, not a chosen
nuclear Fock truncation. No basis, Hamiltonian, regulator or H2 is imported.
Written words act right-to-left. Tensor axes follow the supplied mode order.
"""
from dataclasses import dataclass
from itertools import product
from math import sqrt, prod
from operator import index

import numpy as np


@dataclass(frozen=True)
class Mode:
    label: str
    statistics: str
    capacity: int

    def __post_init__(self):
        if (not self.label or self.statistics not in ('fermion', 'boson')
                or isinstance(self.capacity, bool) or index(self.capacity) < 1
                or (self.statistics == 'fermion' and self.capacity != 1)):
            raise ValueError('Named fermion(capacity 1) or boson(positive capacity) required.')


@dataclass(frozen=True)
class Ladder:
    mode: int
    creation: bool

    def adjoint(self):
        return Ladder(self.mode, not self.creation)


@dataclass(frozen=True)
class Word:
    coefficient: complex
    factors: tuple

    def adjoint(self):
        return Word(np.conj(self.coefficient), tuple(x.adjoint() for x in self.factors[::-1]))


class FockRegister:
    def __init__(self, modes):
        self.modes = tuple(modes)
        if not self.modes or len({x.label for x in self.modes}) != len(self.modes):
            raise ValueError('Nonempty distinctly named modes required.')
        self.dimensions = tuple(x.capacity+1 for x in self.modes)
        self.dimension = prod(self.dimensions)

    def _word(self, word):
        if not np.isfinite(word.coefficient):
            raise ValueError('Finite coefficient required.')
        for op in word.factors:
            if (isinstance(op.mode, bool) or index(op.mode) < 0 or op.mode >= len(self.modes)
                    or not isinstance(op.creation, bool)):
                raise ValueError('Valid mode index and boolean creation flag required.')

    def _state(self, state):
        for occupations, amplitude in state.items():
            if len(occupations) != len(self.modes) or not np.isfinite(amplitude):
                raise ValueError('Finite amplitudes with one occupation per mode required.')
            if any(isinstance(n, bool) or index(n) < 0 or n > m.capacity
                   for n, m in zip(occupations, self.modes)):
                raise ValueError('Input occupation outside the declared register.')

    def occupation_action(self, words, state, *, truncate_each_ladder=True):
        """Independent occupation/sign reference, optionally retaining intermediates.

        With False, boson intermediates and outputs can exceed local capacity.
        Project the final result explicitly. This distinguishes P O P from a
        product of individually projected ladders at a boson boundary.
        """
        self._state(state)
        result = {}
        for word in words:
            self._word(word)
            for initial, amplitude in state.items():
                occupations = list(initial)
                value = complex(word.coefficient)*amplitude
                for op in reversed(word.factors):
                    mode, n = self.modes[op.mode], occupations[op.mode]
                    if mode.statistics == 'fermion':
                        if n == int(op.creation):
                            value = 0j
                            break
                        value *= (-1)**sum(occupations[j] for j in range(op.mode)
                                           if self.modes[j].statistics == 'fermion')
                    else:
                        if not op.creation and n == 0:
                            value = 0j
                            break
                        if op.creation and truncate_each_ladder and n == mode.capacity:
                            value = 0j
                            break
                        value *= sqrt(n+1 if op.creation else n)
                    occupations[op.mode] += 1 if op.creation else -1
                if value != 0:
                    key = tuple(occupations)
                    result[key] = result.get(key, 0j)+value
        return {k: v for k, v in result.items() if v != 0}

    def tensor_factors(self, word):
        """Exact Jordan--Wigner graded product; bosons do not add parity strings."""
        self._word(word)
        factors = [np.eye(d, dtype=complex) for d in self.dimensions]
        for op in word.factors:
            mode = self.modes[op.mode]
            if mode.statistics == 'fermion':
                for j in range(op.mode):
                    if self.modes[j].statistics == 'fermion':
                        factors[j] = factors[j]@np.diag([1., -1.])
            local = np.diag(np.sqrt(np.arange(1, self.dimensions[op.mode])), 1)
            if op.creation:
                local = local.T
            factors[op.mode] = factors[op.mode]@local
        return tuple(factors)

    def tensor_action(self, words, state):
        """Contract product-operator tensors on sparse states without a dense H."""
        self._state(state)
        result = {}
        for word in words:
            tensors = self.tensor_factors(word)
            for initial, amplitude in state.items():
                choices = [tuple((int(i), matrix[i, n]) for i in np.flatnonzero(matrix[:, n]))
                           for matrix, n in zip(tensors, initial)]
                for outcome in product(*choices):
                    key = tuple(x[0] for x in outcome)
                    value = word.coefficient*amplitude*prod(x[1] for x in outcome)
                    result[key] = result.get(key, 0j)+value
        return {k: v for k, v in result.items() if v != 0}

    def project(self, state):
        return {k: v for k, v in state.items()
                if all(0 <= n <= m.capacity for n, m in zip(k, self.modes))}

    def dense(self, words, *, max_dimension=1024):
        if self.dimension > max_dimension:
            raise ValueError('Dense reference exceeds the explicit dimension bound.')
        result = np.zeros((self.dimension, self.dimension), complex)
        for word in words:
            term = np.array([[word.coefficient]], complex)
            for local in self.tensor_factors(word):
                term = np.kron(term, local)
            result += term
        return result

    def matrix_elements(self, words, outgoing, incoming):
        """Selected complete matrix elements, not a claim that the carrier is closed."""
        rows = tuple(outgoing)
        result = np.zeros((len(rows), len(incoming)), complex)
        for j, column in enumerate(incoming):
            transformed = self.tensor_action(words, {tuple(column): 1.})
            result[:, j] = [transformed.get(tuple(row), 0) for row in rows]
        return result

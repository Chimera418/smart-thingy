"""
optimizer.py
Hybrid metaheuristic: GA (global) + VNS (local) + Quantum-inspired mutation.
Solves multi-objective route optimization: minimize (time, distance, congestion).
"""
import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Tuple, Dict, Callable
import copy


# ─── Quantum-Inspired Mutation ───────────────────────────────────────────────
def quantum_mutation(individual: list, prob: float = 0.1) -> list:
    """
    Quantum-inspired mutation: each 'qubit' (position) has a probability
    of flipping based on a quantum rotation angle θ.
    Simulates superposition collapse for route segments.
    """
    ind = list(individual)
    for i in range(len(ind)):
        if random.random() < prob:
            # Quantum rotation: θ controls flip probability
            theta = random.uniform(0, np.pi / 2)
            if random.random() < np.sin(theta) ** 2:
                # Swap with a random other position (simulates entanglement)
                j = random.randint(0, len(ind) - 1)
                ind[i], ind[j] = ind[j], ind[i]
    return ind


# ─── VNS Local Search ────────────────────────────────────────────────────────
def vns_refine(route: list, cost_func: Callable, max_iter: int = 20) -> list:
    """
    Variable Neighborhood Search:
    N1: Swap (2-opt)
    N2: Or-opt (move 1-3 consecutive nodes)
    N3: 3-opt (reverse 3-segment)
    """
    current = list(route)
    best_cost = cost_func(current)

    for _ in range(max_iter):
        improved = False

        # N1: 2-opt swap
        for i in range(len(current) - 1):
            for j in range(i + 1, len(current)):
                candidate = current[:i] + [current[j]] + current[i+1:j] + [current[i]] + current[j+1:]
                # Simpler: just swap positions i and j
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                c = cost_func(candidate)
                if c < best_cost:
                    current = candidate
                    best_cost = c
                    improved = True
                    break
            if improved:
                break

        if not improved:
            # N2: Or-opt (move 1-2 nodes)
            for i in range(len(current)):
                for j in range(len(current)):
                    if i == j or abs(i-j) <= 1:
                        continue
                    candidate = list(current)
                    node = candidate.pop(i)
                    candidate.insert(j, node)
                    c = cost_func(candidate)
                    if c < best_cost:
                        current = candidate
                        best_cost = c
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            # N3: 3-opt reverse
            for i in range(len(current) - 2):
                for j in range(i + 2, len(current)):
                    candidate = list(current)
                    candidate[i:j+1] = candidate[i:j+1][::-1]
                    c = cost_func(candidate)
                    if c < best_cost:
                        current = candidate
                        best_cost = c
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return current


# ─── GA Setup ────────────────────────────────────────────────────────────────
def setup_ga(route_length: int, pop_size: int = 50,
             cx_prob: float = 0.7, mut_prob: float = 0.2,
             ngen: int = 50):
    """Configure DEAP toolbox for route optimization."""

    if not hasattr(creator, 'FitnessMin'):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Individual = permutation of candidate route indices
    toolbox.register('indices', random.sample, range(route_length), route_length)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    return toolbox


def evaluate_routes(population: list, cost_func: Callable):
    """Evaluate fitness for a population of route permutations."""
    for ind in population:
        ind.fitness.values = (cost_func(ind),)


def cx_order(ind1, ind2, prob=0.7):
    """Order crossover (OX1) for permutation encoding."""
    if random.random() > prob:
        return ind1, ind2

    size = len(ind1)
    a, b = sorted(random.sample(range(size), 2))

    # Copy segment from ind1
    child1 = [None] * size
    child1[a:b+1] = ind1[a:b+1]
    fill = [x for x in ind2 if x not in child1[a:b+1]]
    pos = 0
    for i in range(size):
        if child1[i] is None:
            child1[i] = fill[pos]
            pos += 1

    # Copy segment from ind2
    child2 = [None] * size
    child2[a:b+1] = ind2[a:b+1]
    fill = [x for x in ind1 if x not in child2[a:b+1]]
    pos = 0
    for i in range(size):
        if child2[i] is None:
            child2[i] = fill[pos]
            pos += 1

    del ind1.fitness.values
    del ind2.fitness.values
    return child1, child2


# ─── Main Optimizer ──────────────────────────────────────────────────────────
class HybridRouteOptimizer:
    """
    Hybrid GA + VNS + Quantum Mutation for traffic route optimization.
    
    Multi-objective: minimize (travel_time, distance, congestion_exposure)
    """

    def __init__(self, candidate_routes: List[list],
                 cost_func: Callable,
                 pop_size: int = 40,
                 ngen: int = 30,
                 cx_prob: float = 0.7,
                 mut_prob: float = 0.2,
                 vns_iter: int = 15):
        """
        Args:
            candidate_routes: List of K candidate paths (node sequences)
            cost_func: Callable that takes a route index permutation and returns composite cost
            pop_size: GA population size
            ngen: Number of generations
            cx_prob: Crossover probability
            mut_prob: Mutation probability
            vns_iter: VNS iterations per individual
        """
        self.candidate_routes = candidate_routes
        self.cost_func = cost_func
        self.pop_size = pop_size
        self.ngen = ngen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.vns_iter = vns_iter
        self.history = []  # Track best fitness per generation

    def optimize(self) -> Dict:
        """Run the hybrid optimization. Returns best route info."""
        k = len(self.candidate_routes)
        toolbox = setup_ga(k, self.pop_size, self.cx_prob, self.mut_prob, self.ngen)

        # Register operators
        toolbox.register('evaluate', evaluate_routes, cost_func=self.cost_func)
        toolbox.register('select', tools.selTournament, tournsize=3)
        toolbox.register('mate', cx_order, prob=self.cx_prob)
        toolbox.register('mutate', quantum_mutation, prob=self.mut_prob)

        # Initialize population
        pop = toolbox.population(n=self.pop_size)
        toolbox.evaluate(pop)

        # Evolution loop
        for gen in range(self.ngen):
            # Select
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.cx_prob:
                    offspring[i], offspring[i+1] = toolbox.mate(
                        offspring[i], offspring[i+1])
                    del offspring[i].fitness.values
                    del offspring[i+1].fitness.values

            # Mutation (Quantum-inspired)
            for i in range(len(offspring)):
                if random.random() < self.mut_prob:
                    offspring[i] = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # VNS local refinement on top-10%
            top_n = max(1, len(offspring) // 10)
            sorted_off = sorted(offspring, key=lambda x: x.fitness.values[0])
            for ind in sorted_off[:top_n]:
                refined = vns_refine(list(ind), self.cost_func, self.vns_iter)
                ind[:] = refined
                del ind.fitness.values

            # Re-evaluate invalid
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            toolbox.evaluate(invalid)

            # Replace population
            pop[:] = offspring

            # Track history
            best_fit = min(ind.fitness.values[0] for ind in pop)
            self.history.append(best_fit)

        # Extract best
        best_ind = tools.selBest(pop, 1)[0]
        best_route_indices = list(best_ind)

        # Map back to actual route
        best_route = self._decode(best_route_indices)

        return {
            'best_route': best_route,
            'best_cost': best_ind.fitness.values[0],
            'route_indices': best_route_indices,
            'history': self.history,
            'generations': self.ngen
        }

    def _decode(self, indices: list) -> list:
        """Decode permutation of candidate route indices into a single route."""
        # Concatenate candidate routes in the given order (simplified)
        # In practice, this would be a proper route reconstruction
        route = []
        for idx in indices:
            route.extend(self.candidate_routes[idx])
        # Remove consecutive duplicates
        deduped = [route[0]] + [b for a, b in zip(route, route[1:]) if a != b]
        return deduped

    def benchmark(self, algorithms: Dict[str, Callable] = None) -> Dict:
        """
        Run benchmark comparison between algorithms.
        Returns results table for SIH demo.
        """
        results = {}

        # Hybrid (our method)
        hybrid_result = self.optimize()
        results['Hybrid GA+VNS+QM'] = {
            'cost': hybrid_result['best_cost'],
            'time': 0  # would measure actual time
        }

        # Pure GA (no VNS, no quantum)
        if algorithms:
            for name, func in algorithms.items():
                result = func()
                results[name] = result

        return results   
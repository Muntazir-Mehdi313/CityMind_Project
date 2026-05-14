# challenge3/ga_core.py — Genetic Algorithm Core (Challenge 3)
# Other classes used :
#   Uses models.city_node.CityNode  & models.city_graph.CityGraph

import random
from typing import List, Tuple

from models.city_node import CityNode
from models.city_graph import CityGraph
from challenge3.fitness import get_cached_fitness, get_eligible_ambulance_nodes

# GA CONFIGURATION

GA_CONFIG = {
    "population_size":  100,
    "max_generations":  200,
    "tournament_size":  5,
    "crossover_rate":   0.85,
    "mutation_rate":    0.15,
    "elite_count":      2,
    "stagnation_limit": 30,
    "num_ambulances":   3,
}

# POPULATION INITIALISATION
def initialize_population(
        graph: CityGraph,
    population_size: int = 100,
    num_ambulances: int = 3,
) -> List[List[CityNode]]:
    """
    Creates `population_size` random chromosomes, each a list of 3 distinct
    eligible ambulance nodes (Hospital or Ambulance Depot, accessible).
    """
    eligible = get_eligible_ambulance_nodes(graph)

    if len(eligible) < num_ambulances:
        raise ValueError(
            f"Cannot create population: only {len(eligible)} eligible ambulance "
            f"positions on the grid. Need at least {num_ambulances}."
        )

    population = []
    for _ in range(population_size):
        chromosome = random.sample(eligible, num_ambulances)
        population.append(chromosome)

    return population


# SELECTION

def tournament_selection(
        population: List[List[CityNode]],
        fitness_scores: List[float],
        tournament_size: int = 5
) -> List[CityNode]:
    """
    Selects a parent via k-tournament selection.
    Lower fitness (shorter worst-case distance) wins.
    """
    indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    winner_idx = min(indices, key=lambda i: fitness_scores[i])
    return list(population[winner_idx])

# REPAIR

def repair_chromosome(
        chromosome: List[CityNode],
        eligible_nodes: List[CityNode]
) -> List[CityNode]:
    """
    Ensures a chromosome has exactly 3 DISTINCT eligible nodes.
    Duplicates are replaced with random unused eligible nodes.

    Uses id(node) for deduplication (object identity) — correct because
    CityNode.__eq__ is position-based but we want to catch actual
    duplicate object references in the chromosome list.
    """
    seen_ids = set()
    result = []

    for node in chromosome:
        if id(node) not in seen_ids:
            seen_ids.add(id(node))
            result.append(node)

    unused = [n for n in eligible_nodes if id(n) not in seen_ids]
    random.shuffle(unused)

    target_length = len(chromosome)

    while len(result) < target_length:
        if not unused:
            raise RuntimeError(
                "repair_chromosome: not enough eligible nodes to fill chromosome."
            )
        node = unused.pop()
        result.append(node)
        seen_ids.add(id(node))

    return result

# CROSSOVER

def crossover(
        parent1: List[CityNode],
        parent2: List[CityNode],
        eligible_nodes: List[CityNode]
) -> Tuple[List[CityNode], List[CityNode]]:
    """
    Single-point crossover: each child takes one gene from one parent,
    two genes from the other, then repaired for duplicates.
    """
    child1_raw = [parent1[0]] + parent2[1:]
    child2_raw = [parent2[0]] + parent1[1:]

    child1 = repair_chromosome(child1_raw, eligible_nodes)
    child2 = repair_chromosome(child2_raw, eligible_nodes)

    return child1, child2

# MUTATION

def mutate(
        chromosome: List[CityNode],
        eligible_nodes: List[CityNode],
        mutation_rate: float = 0.15
) -> List[CityNode]:
    """
    With probability `mutation_rate`, replaces one random gene with a
    randomly chosen eligible node not already in the chromosome.
    """
    if random.random() > mutation_rate:
        return chromosome

    if not chromosome:
        return chromosome

    idx_to_replace = random.randrange(len(chromosome))
    current_ids = {id(n) for n in chromosome}

    candidates = [n for n in eligible_nodes if id(n) not in current_ids]
    if not candidates:
        return chromosome

    chromosome[idx_to_replace] = random.choice(candidates)
    return chromosome

# ADAPTIVE MUTATION RATE

def adaptive_mutation_rate(
        base_rate: float,
        gens_without_improvement: int,
        stagnation_threshold: int = 20
) -> float:
    """
    Doubles mutation rate when stagnation exceeds threshold, capped at 0.40.
    Prevents premature convergence.
    """
    if gens_without_improvement > stagnation_threshold:
        return min(base_rate * 2.0, 0.40)
    return base_rate

# MAIN GA LOOP

def run_ga(
        graph: CityGraph,
        config: dict = GA_CONFIG
) -> Tuple[List[CityNode], float]:
    """
    Runs the full Genetic Algorithm for ambulance placement.

    Returns:
        best_chromosome : List[CityNode] — 3 nodes where ambulances are placed
        best_fitness    : float — worst-case response distance (minimised)
    """
    print("\n" + "=" * 55)
    print("CHALLENGE 3: Genetic Algorithm — Ambulance Placement")
    print("=" * 55)

    eligible_nodes = get_eligible_ambulance_nodes(graph)
    print(f"  Eligible positions : {len(eligible_nodes)}")
    print(f"  Population         : {config['population_size']} | "
          f"Max generations: {config['max_generations']}")

    # FIX 2: Clear both fields on all nodes
    for node in graph.all_nodes():
        node.ambulance_id   = None
        node.ambulance_here = False

    num_ambulances = config.get("num_ambulances", 3)
    population = initialize_population(
        graph,
        config["population_size"],
        num_ambulances=num_ambulances,
    )

    # FIX 6: Removed redundant pre-evaluation block — gen 1 evaluates population
    best_chromosome: List[CityNode] = []
    best_fitness: float = float('inf')
    gens_no_improve: int = 0

    for generation in range(1, config["max_generations"] + 1):

        fitness_scores = [
            get_cached_fitness(chrom, graph)
            for chrom in population
        ]

        gen_best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        gen_best_fitness = fitness_scores[gen_best_idx]

        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = list(population[gen_best_idx])
            gens_no_improve = 0
        else:
            gens_no_improve += 1

        if generation % 20 == 0 or generation == 1:
            pos_str = str([(n.row, n.col) for n in best_chromosome])
            print(f"  Gen {generation:3d} | Best: {best_fitness:.4f} | "
                  f"No-improve: {gens_no_improve} | Positions: {pos_str}")

        if gens_no_improve >= config["stagnation_limit"]:
            print(f"  [GA] Converged early at generation {generation}.")
            break

        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        next_generation = []

        # Elitism — carry best chromosomes unchanged
        for i in range(config["elite_count"]):
            next_generation.append(list(population[sorted_indices[i]]))

        current_mutation_rate = adaptive_mutation_rate(
            config["mutation_rate"], gens_no_improve
        )

        while len(next_generation) < config["population_size"]:
            p1 = tournament_selection(population, fitness_scores, config["tournament_size"])
            p2 = tournament_selection(population, fitness_scores, config["tournament_size"])

            if random.random() < config["crossover_rate"]:
                c1, c2 = crossover(p1, p2, eligible_nodes)
            else:
                c1, c2 = list(p1), list(p2)

            c1 = mutate(c1, eligible_nodes, current_mutation_rate)
            c2 = mutate(c2, eligible_nodes, current_mutation_rate)

            next_generation.append(c1)
            if len(next_generation) < config["population_size"]:
                next_generation.append(c2)

        population = next_generation

    
    for idx, node in enumerate(best_chromosome):
        node.ambulance_id   = idx   # 0, 1, 2  
        node.ambulance_here = True  # spec boolean field

    print(f"\n  GA Complete.")
    print(f"  Worst-case response distance : {best_fitness:.4f}")
    print(f"  Ambulance positions          : {[(n.row, n.col) for n in best_chromosome]}")
    print("=" * 55 + "\n")

    return best_chromosome, best_fitness
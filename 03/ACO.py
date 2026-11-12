

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# Define cities
NUM_CITIES = 30
CITIES = np.random.uniform(0, 100, (NUM_CITIES, 2))

DISTANCE = np.linalg.norm(
    CITIES[:, None, :] - CITIES[None, :, :], axis=2
)
HEURISTIC = 1.0 / (DISTANCE + 1e-10)
np.fill_diagonal(HEURISTIC, 0)

# ACO parameters
NUM_ANTS = NUM_CITIES
NUM_ITERATIONS = 300
ALPHA = 1.0
BETA = 5.0
RHO = 0.5
Q = 100.0
INITIAL_PHEROMONE = 1.0

def tour_length(tour):
    return sum(
        DISTANCE[tour[i], tour[(i + 1) % len(tour)]]
        for i in range(len(tour))
    )

def nearest_neighbour(start=0):
    unvisited = set(range(NUM_CITIES))
    unvisited.remove(start)
    tour = [start]
    current = start

    while unvisited:
        nxt = min(
            unvisited,
            key=lambda city: DISTANCE[current, city]
        )
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    return tour, tour_length(tour)

def construct_tour(pheromone):
    start = np.random.randint(NUM_CITIES)
    tour = [start]
    unvisited = set(range(NUM_CITIES))
    unvisited.remove(start)
    current = start

    while unvisited:
        candidates = np.array(sorted(unvisited))
        weights = (
            pheromone[current, candidates] ** ALPHA
            * HEURISTIC[current, candidates] ** BETA
        )

        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            probabilities = np.full(
                len(candidates), 1 / len(candidates)
            )
        else:
            probabilities = weights / weights.sum()

        nxt = int(np.random.choice(candidates, p=probabilities))
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    return tour

def ant_colony_optimization():
    pheromone = np.full(
        (NUM_CITIES, NUM_CITIES),
        INITIAL_PHEROMONE,
        dtype=float
    )

    best_tour = None
    best_distance = float("inf")
    convergence = []

    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS):
        all_tours = []

        for _ in range(NUM_ANTS):
            tour = construct_tour(pheromone)
            length = tour_length(tour)
            all_tours.append((tour, length))

            if length < best_distance:
                best_distance = length
                best_tour = tour.copy()

        # Evaporate pheromone
        pheromone *= (1 - RHO)

        # Deposit pheromone on tour edges
        for tour, length in all_tours:
            deposit = Q / length

            for i in range(NUM_CITIES):
                a = tour[i]
                b = tour[(i + 1) % NUM_CITIES]
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        convergence.append(best_distance)

        if (iteration + 1) % 50 == 0:
            print(
                f"Iteration {iteration + 1}: "
                f"Best distance = {best_distance:.2f}"
            )

    runtime = time.perf_counter() - start_time
    return best_tour, best_distance, convergence, runtime

# Run ACO and baseline
nn_tour, nn_distance = nearest_neighbour()

aco_tour, aco_distance, convergence, runtime = (
    ant_colony_optimization()
)

improvement = (
    (nn_distance - aco_distance) / nn_distance
) * 100

print("\n========== RESULTS ==========")
print(f"Cities: {NUM_CITIES}")
print(f"Ants: {NUM_ANTS}")
print(f"Iterations: {NUM_ITERATIONS}")
print(f"Nearest Neighbour distance: {nn_distance:.2f}")
print(f"ACO best distance: {aco_distance:.2f}")
print(f"Improvement over NN: {improvement:.2f}%")
print(f"ACO runtime: {runtime:.3f} seconds")
print("Best route:", aco_tour + [aco_tour[0]])

# Plot best route
def plot_route(tour, title):
    closed_tour = tour + [tour[0]]
    points = CITIES[closed_tour]

    plt.figure(figsize=(8, 6))
    plt.plot(points[:, 0], points[:, 1], "o-")

    for i, (x, y) in enumerate(CITIES):
        plt.annotate(str(i), (x, y), fontsize=8)

    plt.title(title)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

plot_route(
    aco_tour,
    f"ACO Best Route (Distance = {aco_distance:.2f})"
)

# Plot convergence
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, NUM_ITERATIONS + 1),
    convergence,
    label="ACO best distance",
    color="green"
)
plt.axhline(
    nn_distance,
    color="red",
    linestyle="--",
    label="Nearest Neighbour"
)
plt.xlabel("Iteration")
plt.ylabel("Best tour distance")
plt.title("ACO Convergence Analysis")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

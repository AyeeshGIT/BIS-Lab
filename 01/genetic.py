import random
import statistics

# ============================================================
# GENETIC ALGORITHM FOR JOB SCHEDULING
# Objectives:
# 1. Minimize Makespan
# 2. Minimize Workload Imbalance
# ============================================================


# ------------------------------------------------------------
# 1. PROBLEM DATA
# ------------------------------------------------------------

# Processing time required by each job
jobs = {
    "J1": 10,
    "J2": 5,
    "J3": 8,
    "J4": 12,
    "J5": 6,
    "J6": 15,
    "J7": 7,
    "J8": 9,
    "J9": 11,
    "J10": 4
}

# Number of machines
NUM_MACHINES = 3

# GA parameters
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.80

# Weight given to each objective
WEIGHT_MAKESPAN = 0.7
WEIGHT_IMBALANCE = 0.3


# ------------------------------------------------------------
# 2. CREATE INITIAL POPULATION
# ------------------------------------------------------------

def create_individual():
    """
    Create one chromosome.

    Each gene represents the machine assigned to one job.

    Example:
    [0, 1, 2, 0, 1, 2, 1, 0, 2, 1]

    means:
    J1 -> Machine 1
    J2 -> Machine 2
    J3 -> Machine 3
    ...
    """

    return [
        random.randint(0, NUM_MACHINES - 1)
        for _ in range(len(jobs))
    ]


def create_population():
    return [
        create_individual()
        for _ in range(POPULATION_SIZE)
    ]


# ------------------------------------------------------------
# 3. EVALUATE A SCHEDULE
# ------------------------------------------------------------

def calculate_machine_loads(individual):
    """
    Calculate total processing time assigned
    to each machine.
    """

    loads = [0] * NUM_MACHINES

    processing_times = list(jobs.values())

    for job_index, machine in enumerate(individual):
        loads[machine] += processing_times[job_index]

    return loads


def calculate_objectives(individual):
    """
    Returns:
        makespan
        workload imbalance
    """

    loads = calculate_machine_loads(individual)

    # Objective 1: Makespan
    makespan = max(loads)

    # Objective 2: Workload imbalance
    imbalance = statistics.pstdev(loads)

    return makespan, imbalance


# ------------------------------------------------------------
# 4. FITNESS FUNCTION
# ------------------------------------------------------------

def fitness(individual):
    """
    Lower fitness is better.

    We normalize both objectives so that
    one objective does not dominate the other.
    """

    makespan, imbalance = calculate_objectives(individual)

    total_work = sum(jobs.values())

    # Average expected workload
    average_load = total_work / NUM_MACHINES

    normalized_makespan = makespan / average_load
    normalized_imbalance = imbalance / average_load

    # Weighted multi-objective fitness
    score = (
        WEIGHT_MAKESPAN * normalized_makespan
        + WEIGHT_IMBALANCE * normalized_imbalance
    )

    return score


# ------------------------------------------------------------
# 5. SELECTION
# ------------------------------------------------------------

def tournament_selection(population):
    """
    Tournament selection.

    Randomly select two individuals and
    choose the one with better fitness.
    """

    candidate1 = random.choice(population)
    candidate2 = random.choice(population)

    if fitness(candidate1) < fitness(candidate2):
        return candidate1.copy()
    else:
        return candidate2.copy()


# ------------------------------------------------------------
# 6. CROSSOVER
# ------------------------------------------------------------

def crossover(parent1, parent2):
    """
    Single-point crossover.
    """

    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2


# ------------------------------------------------------------
# 7. MUTATION
# ------------------------------------------------------------

def mutation(individual):
    """
    Randomly change the machine assigned to a job.
    """

    mutated = individual.copy()

    for i in range(len(mutated)):

        if random.random() < MUTATION_RATE:

            # Assign job to another random machine
            mutated[i] = random.randint(
                0,
                NUM_MACHINES - 1
            )

    return mutated


# ------------------------------------------------------------
# 8. GENETIC ALGORITHM
# ------------------------------------------------------------

def genetic_algorithm():

    # Create initial population
    population = create_population()

    best_solution = None
    best_fitness = float("inf")

    for generation in range(GENERATIONS):

        # Evaluate population
        population = sorted(
            population,
            key=fitness
        )

        # Store best solution
        if fitness(population[0]) < best_fitness:

            best_solution = population[0].copy()
            best_fitness = fitness(best_solution)

        # Elitism:
        # Keep the best 2 solutions
        new_population = [
            population[0].copy(),
            population[1].copy()
        ]

        # Generate new population
        while len(new_population) < POPULATION_SIZE:

            # Selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutation(child1)
            child2 = mutation(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

        # Display progress
        if generation % 10 == 0:

            makespan, imbalance = calculate_objectives(
                best_solution
            )

            print(
                f"Generation {generation:3d} | "
                f"Makespan = {makespan:5.2f} | "
                f"Imbalance = {imbalance:5.2f} | "
                f"Fitness = {best_fitness:.4f}"
            )

    return best_solution


# ------------------------------------------------------------
# 9. DISPLAY FINAL SOLUTION
# ------------------------------------------------------------

best_solution = genetic_algorithm()

makespan, imbalance = calculate_objectives(best_solution)

loads = calculate_machine_loads(best_solution)

print("\n==========================================")
print("         FINAL OPTIMIZED SCHEDULE")
print("==========================================")

job_names = list(jobs.keys())

for machine in range(NUM_MACHINES):

    print(f"\nMachine {machine + 1}:")

    machine_jobs = []

    for i, assigned_machine in enumerate(best_solution):

        if assigned_machine == machine:

            machine_jobs.append(
                job_names[i]
            )

    print(machine_jobs)

    print(
        "Total workload:",
        loads[machine]
    )


print("\n------------------------------------------")

print("Best Makespan:", makespan)

print("Workload Imbalance:", round(imbalance, 2))

print("Best Fitness:", round(
    fitness(best_solution), 4
))

print("------------------------------------------")

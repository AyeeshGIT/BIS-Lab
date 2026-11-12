
import numpy as np
import math

# ============================================================
# CUCKOO SEARCH FOR CLOUD LOAD BALANCING
# Objective 1: Minimize workload imbalance
# Objective 2: Minimize power consumption
# Compatible with Python 3.14
# ============================================================

np.random.seed(42)

# -------------------- Problem Definition --------------------
NUM_TASKS = 30
NUM_VMS = 5

# Task lengths in Million Instructions (MI)
task_lengths = np.random.randint(
    500, 5000, NUM_TASKS
).astype(float)

# VM processing speeds in Million Instructions per second
vm_mips = np.array(
    [1000, 1200, 1500, 1800, 2000], dtype=float
)

# VM power characteristics in watts
idle_power = np.array(
    [70, 75, 85, 95, 105], dtype=float
)

max_power = np.array(
    [140, 150, 170, 190, 210], dtype=float
)

# Time horizon for estimating CPU utilization
HORIZON_SECONDS = 10.0

# Cuckoo Search parameters
NESTS = 25
MAX_ITERATIONS = 300
PA = 0.25
ALPHA = 0.20

# Objective weights
W_BALANCE = 0.50
W_POWER = 0.50

EPS = 1e-12


# -------------------- Assignment Utilities ------------------
def decode(solution):
    """Convert continuous positions into VM assignments."""
    return np.clip(
        np.floor(solution).astype(int),
        0,
        NUM_VMS - 1
    )


def get_loads(assignment):
    """Calculate the workload assigned to each VM."""
    return np.bincount(
        assignment,
        weights=task_lengths,
        minlength=NUM_VMS
    )


# -------------------- Fitness Evaluation --------------------
def evaluate(assignment):
    """
    Evaluate:
    1. Workload imbalance
    2. Power consumption
    3. Makespan
    """

    loads = get_loads(assignment)

    # Objective 1: Workload imbalance
    mean_load = np.mean(loads)
    imbalance = np.std(loads) / (mean_load + EPS)

    # Estimate execution time for each VM
    execution_times = loads / vm_mips

    # Estimate CPU utilization
    utilization = np.clip(
        execution_times / HORIZON_SECONDS,
        0.0,
        1.0
    )

    # Objective 2: Total estimated power consumption
    power_per_vm = np.where(
        loads > 0,
        idle_power
        + (max_power - idle_power) * utilization,
        0.0
    )

    total_power = np.sum(power_per_vm)

    # Combined fitness: lower is better
    fitness = (
        W_BALANCE * imbalance
        + W_POWER * (total_power / 1000.0)
    )

    # Makespan: maximum estimated VM execution time
    makespan = np.max(execution_times)

    return (
        fitness,
        imbalance,
        total_power,
        makespan,
        loads
    )


# -------------------- Levy Flight ---------------------------
def levy_flight(shape, beta=1.5):
    """
    Generate Levy-flight steps using Mantegna's method.
    Uses Python's math module for Python 3.14 compatibility.
    """

    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma_u, shape)
    v = np.random.normal(0, 1, shape)

    return u / (np.abs(v) ** (1 / beta) + EPS)


# -------------------- Round Robin Baseline -----------------
def round_robin():
    """Assign tasks to VMs in cyclic order."""
    return np.arange(NUM_TASKS) % NUM_VMS


# -------------------- Cuckoo Search Algorithm ---------------
def cuckoo_search():

    # Initialize nests
    nests = np.random.uniform(
        0,
        NUM_VMS,
        size=(NESTS, NUM_TASKS)
    )

    fitness = np.array([
        evaluate(decode(nest))[0]
        for nest in nests
    ])

    # Initialize best solution
    best_index = np.argmin(fitness)
    best_nest = nests[best_index].copy()
    best_fitness = fitness[best_index]

    history = []

    for iteration in range(MAX_ITERATIONS):

        # Generate new solutions using Levy flights
        for i in range(NESTS):

            step = levy_flight(NUM_TASKS)

            candidate = nests[i] + ALPHA * step * (
                nests[i] - best_nest
            )

            candidate = np.clip(
                candidate,
                0,
                NUM_VMS - 1
            )

            candidate_fitness = evaluate(
                decode(candidate)
            )[0]

            # Compare against a randomly selected nest
            j = np.random.randint(NESTS)

            if candidate_fitness < fitness[j]:
                nests[j] = candidate.copy()
                fitness[j] = candidate_fitness

        # Abandon a fraction of the worst nests
        num_abandoned = max(1, int(PA * NESTS))

        worst_indices = np.argsort(fitness)[
            -num_abandoned:
        ]

        for i in worst_indices:
            nests[i] = np.random.uniform(
                0,
                NUM_VMS,
                NUM_TASKS
            )

            fitness[i] = evaluate(
                decode(nests[i])
            )[0]

        # Update the best solution
        best_index = np.argmin(fitness)

        if fitness[best_index] < best_fitness:
            best_fitness = fitness[best_index]
            best_nest = nests[best_index].copy()

        history.append(best_fitness)

        if (iteration + 1) % 50 == 0:
            print(
                f"Iteration {iteration + 1:3d} | "
                f"Best fitness: {best_fitness:.4f}"
            )

    return decode(best_nest), history


# -------------------- Main Program --------------------------
if __name__ == "__main__":

    print("=" * 55)
    print("CUCKOO SEARCH FOR CLOUD LOAD BALANCING")
    print("=" * 55)

    # Round Robin solution
    rr_assignment = round_robin()
    rr = evaluate(rr_assignment)

    # Cuckoo Search solution
    cs_assignment, history = cuckoo_search()
    cs = evaluate(cs_assignment)

    # Display Round Robin results
    print("\n========== ROUND ROBIN BASELINE ==========")
    print(f"Workload imbalance : {rr[1]:.4f}")
    print(f"Estimated power    : {rr[2]:.2f} W")
    print(f"Makespan           : {rr[3]:.2f} seconds")
    print(f"VM workloads (MI)  : {rr[4]}")

    # Display Cuckoo Search results
    print("\n========== CUCKOO SEARCH RESULTS ==========")
    print(f"Workload imbalance : {cs[1]:.4f}")
    print(f"Estimated power    : {cs[2]:.2f} W")
    print(f"Makespan           : {cs[3]:.2f} seconds")
    print(f"VM workloads (MI)  : {cs[4]}")
    print(f"Best fitness       : {cs[0]:.4f}")

    # Display task assignments
    print("\n========== TASK ASSIGNMENTS ==========")

    for task_id, vm_id in enumerate(cs_assignment):
        print(f"Task {task_id:02d} -> VM {vm_id}")

    # Calculate relative improvement over Round Robin
    balance_improvement = (
        (rr[1] - cs[1]) / (rr[1] + EPS)
    ) * 100

    power_improvement = (
        (rr[2] - cs[2]) / (rr[2] + EPS)
    ) * 100

    print("\n========== PERFORMANCE COMPARISON ==========")
    print(
        f"Imbalance reduction: "
        f"{balance_improvement:.2f}%"
    )

    print(
        f"Power reduction    : "
        f"{power_improvement:.2f}%"
    )

    print("\nExecution completed successfully.")

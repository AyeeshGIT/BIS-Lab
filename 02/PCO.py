import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------
# 1. POWER SYSTEM DATA (ILLUSTRATIVE)
# ------------------------------------------------
N = 3
DEMAND = 300.0  # MW

P_MIN = np.array([50.0, 50.0, 50.0])
P_MAX = np.array([200.0, 150.0, 100.0])

# Fuel cost: a + b*P + c*P^2 ($/h)
a = np.array([500.0, 400.0, 200.0])
b = np.array([5.3, 5.5, 5.8])
c = np.array([0.004, 0.006, 0.009])

# Emission: alpha + beta*P + gamma*P^2 (kg/h)
alpha = np.array([100.0, 120.0, 150.0])
beta = np.array([0.20, 0.18, 0.15])
gamma = np.array([0.0004, 0.0005, 0.0006])

# PSO parameters
SWARM_SIZE = 50
ITERATIONS = 300
W = 0.7
C1 = 1.5
C2 = 1.5


# ------------------------------------------------
# 2. OBJECTIVE FUNCTIONS
# ------------------------------------------------
def fuel_cost(P):
    return np.sum(a + b * P + c * P**2)


def emissions(P):
    return np.sum(alpha + beta * P + gamma * P**2)


# ------------------------------------------------
# 3. REPAIR POWER BALANCE AND LIMIT CONSTRAINTS
# ------------------------------------------------
def repair(P):
    P = np.clip(P, P_MIN, P_MAX)

    # Redistribute the power mismatch while respecting
    # generator minimum and maximum limits.
    for _ in range(20):
        mismatch = DEMAND - np.sum(P)

        if abs(mismatch) < 1e-8:
            break

        if mismatch > 0:
            capacity = P_MAX - P
        else:
            capacity = P - P_MIN

        available = np.sum(capacity)

        if available < 1e-10:
            break

        change = min(abs(mismatch), available)
        P += np.sign(mismatch) * capacity * change / available
        P = np.clip(P, P_MIN, P_MAX)

    return P


# Normalize objectives to comparable scales
C_SCALE = max(
    fuel_cost(P_MAX) - fuel_cost(P_MIN), 1.0
)
E_SCALE = max(
    emissions(P_MAX) - emissions(P_MIN), 1.0
)


# ------------------------------------------------
# 4. PARTICLE SWARM OPTIMIZATION
# ------------------------------------------------
def run_pso(mode="compromise"):
    position = np.random.uniform(
        P_MIN, P_MAX, (SWARM_SIZE, N)
    )

    for i in range(SWARM_SIZE):
        position[i] = repair(position[i])

    velocity = np.zeros_like(position)
    pbest = position.copy()

    def fitness(P):
        if mode == "cost":
            return fuel_cost(P) / C_SCALE

        if mode == "emission":
            return emissions(P) / E_SCALE

        # Equal-weight compromise between both objectives
        return (
            0.5 * fuel_cost(P) / C_SCALE
            + 0.5 * emissions(P) / E_SCALE
        )

    pbest_score = np.array([fitness(p) for p in pbest])
    best_idx = np.argmin(pbest_score)
    gbest = pbest[best_idx].copy()
    gbest_score = pbest_score[best_idx]

    history = []

    for iteration in range(ITERATIONS):
        r1 = np.random.rand(SWARM_SIZE, N)
        r2 = np.random.rand(SWARM_SIZE, N)

        velocity = (
            W * velocity
            + C1 * r1 * (pbest - position)
            + C2 * r2 * (gbest - position)
        )

        max_velocity = 0.2 * (P_MAX - P_MIN)
        velocity = np.clip(
            velocity, -max_velocity, max_velocity
        )

        position = position + velocity

        for i in range(SWARM_SIZE):
            position[i] = repair(position[i])

        scores = np.array([fitness(p) for p in position])

        improved = scores < pbest_score
        pbest[improved] = position[improved]
        pbest_score[improved] = scores[improved]

        best_idx = np.argmin(pbest_score)

        if pbest_score[best_idx] < gbest_score:
            gbest_score = pbest_score[best_idx]
            gbest = pbest[best_idx].copy()

        history.append(gbest_score)

    return gbest, history


# ------------------------------------------------
# 5. SOLVE BOTH OBJECTIVES AND COMPROMISE
# ------------------------------------------------
cost_solution, cost_history = run_pso("cost")
emission_solution, emission_history = run_pso("emission")
compromise_solution, compromise_history = run_pso(
    "compromise"
)


# ------------------------------------------------
# 6. DISPLAY RESULTS
# ------------------------------------------------
solutions = [
    ("Minimum fuel cost", cost_solution),
    ("Minimum emissions", emission_solution),
    ("Balanced compromise", compromise_solution),
]

print("\n========== PSO DISPATCH RESULTS ==========")

for name, P in solutions:
    print(f"\n{name}")
    for i, power in enumerate(P):
        print(f"Generator {i + 1}: {power:.2f} MW")

    print(f"Total generation: {np.sum(P):.2f} MW")
    print(f"Fuel cost: ${fuel_cost(P):.2f}/h")
    print(f"Emissions: {emissions(P):.2f} kg/h")
    print(f"Power balance error: {np.sum(P) - DEMAND:.8f} MW")


# ------------------------------------------------
# 7. CONVERGENCE GRAPH
# ------------------------------------------------
plt.figure(figsize=(9, 5))

plt.plot(cost_history, label="Minimum cost")
plt.plot(emission_history, label="Minimum emissions")
plt.plot(compromise_history, label="Compromise")

plt.xlabel("Iteration")
plt.ylabel("Normalized fitness")
plt.title("PSO Convergence for Economic Emission Dispatch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ------------------------------------------------
# 8. COMPARE OBJECTIVE VALUES
# ------------------------------------------------
names = [name for name, _ in solutions]
costs = [fuel_cost(P) for _, P in solutions]
emission_values = [emissions(P) for _, P in solutions]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(names, costs, color="steelblue")
axes[0].set_title("Fuel Cost Comparison")
axes[0].set_ylabel("Cost ($/h)")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(names, emission_values, color="darkorange")
axes[1].set_title("Emission Comparison")
axes[1].set_ylabel("Emissions (kg/h)")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()
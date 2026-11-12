
import random
import copy
import operator
import math
import numpy as np
import matplotlib.pyplot as plt

from deap import base, creator, tools, gp, algorithms

# ==========================================
# 1. DATASET FOR SYMBOLIC REGRESSION
# ==========================================

random.seed(42)
np.random.seed(42)

X = np.linspace(-5, 5, 150)

# Target mathematical function
Y = X**2 + np.sin(X)

# Training and testing data
X_train = X[:110]
Y_train = Y[:110]

X_test = X[110:]
Y_test = Y[110:]

# ==========================================
# 2. DEFINE MATHEMATICAL OPERATORS
# ==========================================

def protected_div(a, b):
    if abs(b) < 1e-8:
        return 1.0
    return a / b

pset = gp.PrimitiveSet("MAIN", 1)

pset.renameArguments(ARG0="x")

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)

pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)

pset.addEphemeralConstant(
    "constant",
    lambda: random.uniform(-5, 5)
)

# ==========================================
# 3. FITNESS AND INDIVIDUAL
# ==========================================

if not hasattr(creator, "FitnessMin"):
    creator.create(
        "FitnessMin",
        base.Fitness,
        weights=(-1.0,)
    )

if not hasattr(creator, "Individual"):
    creator.create(
        "Individual",
        gp.PrimitiveTree,
        fitness=creator.FitnessMin
    )

toolbox = base.Toolbox()

toolbox.register(
    "expr",
    gp.genHalfAndHalf,
    pset=pset,
    min_=1,
    max_=3
)

toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    toolbox.expr
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

toolbox.register(
    "compile",
    gp.compile,
    pset=pset
)

# ==========================================
# 4. FITNESS EVALUATION
# ==========================================

def evaluate(individual):
    try:
        func = toolbox.compile(expr=individual)

        prediction = np.array(
            [func(x) for x in X_train],
            dtype=float
        )

        if not np.all(np.isfinite(prediction)):
            return (1e10,)

        error = np.sqrt(
            np.mean((Y_train - prediction)**2)
        )

        return (float(error),)

    except Exception:
        return (1e10,)

toolbox.register("evaluate", evaluate)

toolbox.register(
    "mate",
    gp.cxOnePoint
)

toolbox.register(
    "expr_mut",
    gp.genFull,
    min_=0,
    max_=2
)

toolbox.register(
    "mutate",
    gp.mutUniform,
    expr=toolbox.expr_mut,
    pset=pset
)

toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=8
    )
)

toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=operator.attrgetter("height"),
        max_value=8
    )
)

# ==========================================
# 5. CELLULAR NEIGHBOURHOOD
# ==========================================

def get_neighbors(index, radius=1):

    row = index // 10
    col = index % 10

    neighbors = []

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):

            r = (row + dr) % 10
            c = (col + dc) % 10

            neighbors.append(r * 10 + c)

    return neighbors


def select_parent(population, index, cellular):

    if cellular:
        candidates = [
            population[i]
            for i in get_neighbors(index)
        ]
    else:
        candidates = population

    return copy.deepcopy(
        min(candidates, key=lambda ind: ind.fitness.values[0])
    )

# ==========================================
# 6. PARALLEL CELLULAR GP EXPERIMENT
# ==========================================

def run_experiment(
    cellular=True,
    pipeline="combined",
    generations=100
):

    population = toolbox.population(n=100)

    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    best_history = []
    size_history = []

    for generation in range(generations):

        old_population = copy.deepcopy(population)
        new_population = []

        # Elitism
        elite = copy.deepcopy(
            min(
                old_population,
                key=lambda ind: ind.fitness.values[0]
            )
        )

        for i in range(100):

            parent1 = select_parent(
                old_population, i, cellular
            )

            parent2 = select_parent(
                old_population, i, cellular
            )

            child = copy.deepcopy(parent1)

            # Crossover only
            if pipeline == "crossover":

                child,_ = toolbox.mate(
                    child, parent2
                )

            # Mutation only
            elif pipeline == "mutation":

                child, = toolbox.mutate(child)

            # Combined crossover and mutation
            elif pipeline == "combined":

                if random.random() < 0.8:
                    child, = toolbox.mutate(child)
                else:
                    child, _ = toolbox.mate(
                        child, parent2
                    )

            if child.fitness.valid:
                del child.fitness.values

            toolbox.evaluate(child)
            child.fitness.values = toolbox.evaluate(child)

            new_population.append(child)

        # Preserve elite
        new_population[0] = elite

        population = new_population

        best = min(
            population,
            key=lambda ind: ind.fitness.values[0]
        )

        best_history.append(best.fitness.values[0])
        size_history.append(len(best))

        if generation % 10 == 0:
            print(
                f"Generation {generation}: "
                f"RMSE={best.fitness.values[0]:.5f}, "
                f"Size={len(best)}"
            )

    best = min(
        population,
        key=lambda ind: ind.fitness.values[0]
    )

    return best, best_history, size_history

# ==========================================
# 7. RUN EXPERIMENTS
# ==========================================

results = {}

configurations = [
    ("Global", False, "combined"),
    ("Cellular", True, "combined"),
    ("Cellular Mutation", True, "mutation"),
    ("Cellular Crossover", True, "crossover"),
    ("Cellular Combined", True, "combined")
]

for name, cellular, pipeline in configurations:

    print("\nRunning:", name)

    best, errors, sizes = run_experiment(
        cellular=cellular,
        pipeline=pipeline
    )

    results[name] = {
        "best": best,
        "errors": errors,
        "sizes": sizes
    }

# ==========================================
# 8. TEST AND PRINT RESULTS
# ==========================================

print("\n========== FINAL RESULTS ==========")

for name, result in results.items():

    best = result["best"]

    func = toolbox.compile(expr=best)

    predictions = np.array(
        [func(x) for x in X_test],
        dtype=float
    )

    test_rmse = np.sqrt(
        np.mean((Y_test - predictions)**2)
    )

    print("\nAlgorithm:", name)
    print("Best expression:", best)
    print("Test RMSE:", round(test_rmse, 6))
    print("Expression size:", len(best))

# ==========================================
# 9. OBJECTIVE 1: ACCURACY COMPARISON
# ==========================================

plt.figure(figsize=(10, 5))

for name, result in results.items():

    plt.plot(
        result["errors"],
        label=name
    )

plt.xlabel("Generation")
plt.ylabel("Training RMSE")
plt.title("Objective 1: Symbolic Regression Accuracy")
plt.legend()
plt.grid()
plt.show()

# ==========================================
# 10. OBJECTIVE 2: COMPLEXITY COMPARISON
# ==========================================

plt.figure(figsize=(10, 5))

for name, result in results.items():

    plt.plot(
        result["sizes"],
        label=name
    )

plt.xlabel("Generation")
plt.ylabel("Expression Tree Nodes")
plt.title("Objective 2: Expression Complexity")
plt.legend()
plt.grid()
plt.show()

# ==========================================
# 11. ACTUAL VS PREDICTED OUTPUT
# ==========================================

best = results["Cellular Combined"]["best"]

func = toolbox.compile(expr=best)

predictions = np.array(
    [func(x) for x in X_test]
)

plt.figure(figsize=(10, 5))

plt.scatter(
    X_test,
    Y_test,
    label="Actual values"
)

plt.plot(
    X_test,
    predictions,
    color="red",
    label="Predicted values"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Actual vs Predicted Symbolic Regression")
plt.legend()
plt.grid()
plt.show()

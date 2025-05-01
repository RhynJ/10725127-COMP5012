from deap import base, creator, tools, algorithms
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap.tools import HallOfFame 
from deap.tools import HallOfFame, sortNondominated
from deap.tools.emo import assignCrowdingDist
from scipy.spatial import distance
import winsound # used to wake me up when code is done


# loading data
def load_vrp_data(file_path):
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['Index', 'X', 'Y', 'Demand'])
    return df

# this will need to be changed to run the files needed for optimisation 
file_path = r"F:\\Uni\\Semester 1\\ELEC520\\Optimisation-COMP5012\\CW\\dataSets\\vrp8.txt"
vrp_data = load_vrp_data(file_path)

# remap city IDs to 0-based indexing
original_ids = vrp_data["Index"].tolist()
id_mapping = {old_id: new_id for new_id, old_id in enumerate(original_ids)}
reverse_id_mapping = {v: k for k, v in id_mapping.items()}
vrp_data["Index"] = vrp_data["Index"].map(id_mapping)
city_ids = vrp_data["Index"].tolist()

# Parameters
NUM_DAYS = 3
POP_SIZE = 500
GENERATIONS = 60
geneMutationChance = 0.7

# C + M <= 1 
corssoverProb = 0.6
MutationProb = 0.4

hof = HallOfFame(maxsize=50)

# distance and demand evaluation
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_route_distance(route, coords):
    total_distance = 0
    for i in range(len(route) - 1):
        x1, y1 = coords[route[i]]
        x2, y2 = coords[route[i+1]]
        total_distance += euclidean_distance(x1, y1, x2, y2)
    if route:
        total_distance += euclidean_distance(*coords[route[0]], *coords[route[-1]])
    return total_distance

def calculate_schedule_distance(schedule, coords):
    return sum(calculate_route_distance(day, coords) for day in schedule.values())

def calculate_demand_balance(schedule, demands):
    day_totals = [sum(demands.get(city, 0) for city in route) for route in schedule.values()]
    total = sum(day_totals)
    return (max(day_totals) - min(day_totals)) / (total + 1e-6) if total > 0 else 0

# individual decoding
def decode(individual, num_days):
    schedule = {f"Day {i+1}": [] for i in range(num_days)}
    for i, city in enumerate(individual):
        schedule[f"Day {(i % num_days) + 1}"].append(city)
    return schedule

# DEAP Setup
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("individual", lambda: creator.Individual(random.sample(city_ids, len(city_ids))))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

city_coords = {row.Index: (row.X, row.Y) for row in vrp_data.itertuples(index=False)}
city_demands = {row.Index: row.Demand for row in vrp_data.itertuples(index=False)}


def evaluate(individual):
    schedule = decode(individual, NUM_DAYS)
    total_distance = calculate_schedule_distance(schedule, city_coords)
    demand_balance = calculate_demand_balance(schedule, city_demands)
    return total_distance, demand_balance

# set up NSGA-II
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=geneMutationChance)
toolbox.register("select", tools.selNSGA2)


# MUTAITON TEST: visual proof
print("\nmutation operator verification test")
sample_ind = toolbox.individual()
original = sample_ind[:]
mutated, = toolbox.mutate(toolbox.clone(sample_ind))

print("before mutation:", original)
print("\n\nafter mutation:", mutated)
print("\n\n")
# Check validity
assert sorted(original) == sorted(mutated), " Mutation broke chromosome validity (cities missing or duplicated)"
print(" Mutation preserves chromosome validity.")



# run NSGA-II 
pop = toolbox.population(n=POP_SIZE)
pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=POP_SIZE, lambda_=POP_SIZE, cxpb=corssoverProb, mutpb=MutationProb, ngen=GENERATIONS, stats=None, halloffame=hof, verbose=True)


# evaluate initial population
for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

generational_fronts = []

for gen in range(GENERATIONS):
    print(f"Generation {gen}")

    # variation: crossover and mutation
    offspring = algorithms.varAnd(pop, toolbox, cxpb=corssoverProb, mutpb=MutationProb)

    # evaluate offspring
    for ind in offspring:
        ind.fitness.values = toolbox.evaluate(ind)

    # combine parent + offspring and select next generation
    combined = pop + offspring
    pop = toolbox.select(combined, k=POP_SIZE)

    # store current generation's first front
    front = sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    front_vals = [ind.fitness.values for ind in front]
    generational_fronts.append(front_vals)

    # Update Hall of Fame
    hof.update(pop)

# USED FOR TESTING GENERATIONAL DISTANCE
"""""
# Use final front as reference
reference_front = generational_fronts[-1]

def generational_distance(current_front, reference_front):
    total_dist = 0
    for ind in current_front:
        dists = [distance.euclidean(ind, ref) for ref in reference_front]
        total_dist += min(dists)**2
    return (total_dist / len(current_front))**0.5

gd_values = []
for front in generational_fronts:
    gd = generational_distance(front, reference_front)
    gd_values.append(gd)

# Plot GD over time
plt.figure()
plt.plot(gd_values, marker='o')
plt.title("Generational Distance Over Time")
plt.xlabel("Generation")
plt.ylabel("Generational Distance")
plt.grid(True)
plt.show()
"""

# crowding distance tracking for first Pareto front 
print("\n--- Crowding Distance Analysis ---")
front = sortNondominated(pop, k=len(pop), first_front_only=True)[0]
assignCrowdingDist(front)

# extract and display crowding distances
crowding_distances = [ind.fitness.crowding_dist for ind in front]
for i, dist in enumerate(crowding_distances[:10]):  # Show first 10 for readability
    print(f"Individual {i}: Crowding Distance = {dist:.4f}")

# replace infinite values with estimated high value for plotting
finite_distances = [cd for cd in crowding_distances if np.isfinite(cd)]

# estimate replacement value
if finite_distances:
    replacement_value = max(finite_distances) * 1.2  # or use mean(finite_distances)
else:
    replacement_value = 1.0  # fallback

estimated_distances = [cd if np.isfinite(cd) else replacement_value for cd in crowding_distances]

# Plot with estimated values
plt.figure()
plt.plot(estimated_distances, marker='o')
plt.title("Crowding Distance Across First Pareto Front (with Estimates)")
plt.xlabel("Individual Index (within Front 0)")
plt.ylabel("Crowding Distance")
plt.grid(True)
plt.show()


# select best individual by summed fitness
best_ind = min(pop, key=lambda ind: sum(ind.fitness.values))
print(f"Best individual (by sum of fitness): {best_ind.fitness.values}")

# TESTS
def compute_hypervolume_2d(pareto_front, reference_point):
    """compute the hypervolume using the pareto front with respect to the reference point"""
    # Sort by first objective (total distance) ascending
    pareto_sorted = sorted(pareto_front, key=lambda x: x[0])
    hypervolume = 0.0
    prev_x = reference_point[0]

    for dist, imbalance in reversed(pareto_sorted):  # iterate from worst to best
        width = prev_x - dist
        height = reference_point[1] - imbalance
        if width > 0 and height > 0:
            hypervolume += width * height
        prev_x = dist

    return hypervolume


# access and print Hall of Fame (archive) 
print("\nArchive (Hall of Fame) Individuals:")
for ind in hof:
    print(f"Fitness: {ind.fitness.values}")

# Visualise Pareto Front 
front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
pareto_vals = [ind.fitness.values for ind in front]
pareto_array = np.array(pareto_vals)

# define reference point worse than any solution
ref_point = [max(p[0] for p in pareto_vals) + 10, 1.0] # this can be increase to make sure it is worse 

# Compute hypervolume
hv = compute_hypervolume_2d(pareto_vals, ref_point)
print(f"Hypervolume: {hv:.4f}")

plt.figure()
plt.scatter(pareto_array[:, 0], pareto_array[:, 1])
plt.xlabel("Total Distance")
plt.ylabel("Demand Balance")
plt.title("Pareto Front: Distance vs Demand Balance")
plt.grid(True)

hof_vals = [ind.fitness.values for ind in hof]
hof_array = np.array(hof_vals)

plt.figure()
plt.scatter(hof_array[:, 0], hof_array[:, 1], color='red', label="Hall of Fame")
plt.scatter(pareto_array[:, 0], pareto_array[:, 1], alpha=0.3, label="Final Population")
plt.xlabel("Total Distance")
plt.ylabel("Demand Balance")
plt.title("Final Archive vs Population")
plt.legend()
plt.grid(True)

# visualise Best Route
best_ind = hof[0]
schedule = decode(best_ind, NUM_DAYS)

colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

for i, (day, route) in enumerate(schedule.items()):
    plt.figure(figsize=(8, 6))
    coords = [city_coords[city] for city in route]
    if coords:
        xs, ys = zip(*coords)
        colour = colours[i % len(colours)]
        plt.plot(xs, ys, f'{colour}-o', label=day)
        plt.plot([xs[-1], xs[0]], [ys[-1], ys[0]], f'{colour}--')
    
    plt.title(f"{day} Route")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    #winsound.Beep(1000,1000) # use me to wake you up
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load VRP data
def load_vrp_data(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None, names=['Index', 'X', 'Y', 'Demand'])
    return df

# Load dataset
file_path = r"F:\Uni\Semester 1\ELEC520\Optimisation-COMP5012\CW\dataSets\vrp9.txt"
vrp_data = load_vrp_data(file_path)

# Plot the city locations
def plot_vrp_map(vrp_data, title="VRP City Locations"):
    plt.figure(figsize=(8, 6))
    plt.scatter(vrp_data["X"], vrp_data["Y"], s=vrp_data["Demand"] / 2, c="blue", alpha=0.6, edgecolors="k")

    # Annotate city indices
    for i, row in vrp_data.iterrows():
        plt.annotate(int(row["Index"]), (row["X"], row["Y"]), fontsize=8, ha="right")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(title)
    plt.grid(True)
    plt.show()


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def nearest_neighbour(vrp_data):
    """Generates a greedy TSP route using the nearest neighbour heuristic."""
    unvisited = vrp_data["Index"].tolist()
    current_city = unvisited.pop(0)  # Start from the first city
    route = [current_city]

    while unvisited:
        last_city = vrp_data[vrp_data["Index"] == current_city]
        min_distance = float("inf")
        next_city = None

        for city in unvisited:
            city_data = vrp_data[vrp_data["Index"] == city]
            distance = euclidean_distance(last_city["X"].values[0], last_city["Y"].values[0],
                                          city_data["X"].values[0], city_data["Y"].values[0])
            if distance < min_distance:
                min_distance = distance
                next_city = city

        route.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return route

# Run the nearest neighbour algorithm
test_route = nearest_neighbour(vrp_data)
print("Nearest Neighbour Route:", test_route)


# Display the map
plot_vrp_map(vrp_data)


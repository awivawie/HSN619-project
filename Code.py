import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde

class Agent:
    def __init__(self, id, opinion):
        self.id = id
        self.opinion = opinion
    
    def update_opinion(self, neighbors, homophily_factor):
        similar_neighbors = [n.opinion for n in neighbors if abs(n.opinion - self.opinion) < homophily_factor]
        if similar_neighbors:
            self.opinion += 0.1 * (np.mean(similar_neighbors) - self.opinion)
            self.opinion = max(-1, min(1, self.opinion))


class EchoChamberModel:
    def __init__(self, num_agents, initial_links, homophily_factor):
        self.num_agents = num_agents
        self.homophily_factor = homophily_factor
        self.G = nx.gnm_random_graph(num_agents, initial_links)
        self.agents = {i: Agent(i, np.random.uniform(-1, 1)) for i in range(num_agents)}
        
        for i in self.G.nodes:
            self.G.nodes[i]['opinion'] = self.agents[i].opinion

        self.opinion_variance = []
        self.diversity_index = []
        self.rewiring_counts = []
        self.cluster_sizes = []

    def step(self):
        rewiring_count = 0
        for i in self.G.nodes:
            neighbors = [self.agents[j] for j in self.G.neighbors(i)]
            self.agents[i].update_opinion(neighbors, self.homophily_factor)
            self.G.nodes[i]['opinion'] = self.agents[i].opinion

        for i in self.G.nodes:
            if np.random.rand() < 0.1:
                neighbors = list(self.G.neighbors(i))
                if neighbors:
                    most_dissimilar_neighbor = max(neighbors, key=lambda x: abs(self.agents[x].opinion - self.agents[i].opinion))
                    self.G.remove_edge(i, most_dissimilar_neighbor)
                    
                    potential_new_neighbors = [j for j in self.G.nodes if j != i and j not in neighbors]
                    if potential_new_neighbors:
                        new_neighbor = min(potential_new_neighbors, key=lambda x: abs(self.agents[x].opinion - self.agents[i].opinion))
                        self.G.add_edge(i, new_neighbor)
                        rewiring_count += 1

        opinions = [agent.opinion for agent in self.agents.values()]
        self.opinion_variance.append(np.var(opinions))
        kde = gaussian_kde(opinions)
        density = kde(np.linspace(-1, 1, 200))
        self.diversity_index.append(entropy(density + 1e-10, base=2))  # Adding small value for stability
        self.rewiring_counts.append(rewiring_count)
        self.cluster_sizes.append(len(list(nx.connected_components(self.G.to_undirected()))))

    def run(self, steps):
        for _ in range(steps):
            self.step()

    def plot_opinion_variance(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.opinion_variance, label='Opinion Variance Over Time')
        plt.title("Opinion Variance Across Agents")
        plt.xlabel("Time Step")
        plt.ylabel("Variance")
        plt.legend()
        plt.show()

    def plot_diversity_index(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.diversity_index, label='Diversity Index (Entropy) Over Time', color='purple')
        plt.title("Diversity of Opinions Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Diversity (Entropy)")
        plt.legend()
        plt.show()

    def plot_rewiring_counts(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewiring_counts, label='Rewiring Events Per Step', color='green')
        plt.title("Rewiring Events Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Rewiring Count")
        plt.legend()
        plt.show()
        
    def plot_cluster_size_distribution(self):
        plt.figure(figsize=(10, 5))
        plt.hist(self.cluster_sizes, bins=10, color='orange', edgecolor='black')
        plt.title("Distribution of Cluster Sizes")
        plt.xlabel("Cluster Size")
        plt.ylabel("Frequency")
        plt.show()
        
    def plot_final_opinion_distribution(self):
        opinions = [self.agents[i].opinion for i in self.G.nodes]
        plt.figure(figsize=(10, 6))
        plt.hist(opinions, bins=20, color='skyblue', edgecolor='black')
        plt.title("Final Opinion Distribution After Simulation")
        plt.xlabel("Opinion")
        plt.ylabel("Frequency")
        plt.show()

    def plot_network(self, title="Network Structure"):
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(self.G, seed=42)
        opinions = [self.agents[i].opinion for i in self.G.nodes]
        
        nodes = nx.draw_networkx_nodes(self.G, pos, node_color=opinions, cmap=plt.cm.coolwarm, node_size=50)
        nx.draw_networkx_edges(self.G, pos, alpha=0.5)
        
        colorbar = plt.colorbar(nodes, orientation="horizontal", pad=0.05)
        colorbar.set_label("Opinion")
        plt.title(title)
        plt.show()

num_agents = 100
initial_links = 300
homophily_factor = 0.5
steps = 100

model = EchoChamberModel(num_agents, initial_links, homophily_factor)
model.plot_network("Initial Network Structure")
model.run(steps)
model.plot_opinion_variance()
model.plot_diversity_index()
model.plot_rewiring_counts()
model.plot_cluster_size_distribution()
model.plot_final_opinion_distribution()
model.plot_network("Final Network Structure at Stationary State")

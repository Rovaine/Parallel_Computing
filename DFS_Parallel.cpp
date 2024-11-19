#include <iostream>
#include <vector>
#include <omp.h>
#include <ctime>

class Graph {
public:
    Graph(int vertices);
    void addEdge(int v, int w);
    void dfs(int startVertex);

private:
    int vertices; // Number of vertices
    std::vector<std::vector<int>> adj; // Adjacency list
    void dfsUtil(int vertex, std::vector<bool>& visited);
};

Graph::Graph(int vertices) {
    this->vertices = vertices;
    adj.resize(vertices);
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w); // Add w to v’s list
}

void Graph::dfsUtil(int vertex, std::vector<bool>& visited) {
    visited[vertex] = true;
    std::cout << vertex << " ";

    // Create a local copy of the neighbors to iterate over
    std::vector<int> neighbors = adj[vertex];
    
    // Process neighbors in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < neighbors.size(); i++) {
        int neighbor = neighbors[i];
        if (!visited[neighbor]) {
            // Use a critical section to avoid multiple threads accessing visited array simultaneously
            #pragma omp critical
            {
                if (!visited[neighbor]) {
                    dfsUtil(neighbor, visited);
                }
            }
        }
    }
}

void Graph::dfs(int startVertex) {
    std::vector<bool> visited(vertices, false);
    dfsUtil(startVertex, visited);
}

int main() {
    const int V = 6; // Number of vertices
    Graph graph(V);

    // Creating a sample graph
    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    graph.addEdge(1, 4);
    graph.addEdge(2, 5);

    // Measure execution time
    double startTime = omp_get_wtime();
    
    std::cout << "DFS starting from vertex 0: ";
    graph.dfs(0);
    std::cout << "\n";

    double endTime = omp_get_wtime();
    std::cout << "Execution time: " << (endTime - startTime) << " seconds" << std::endl;

    return 0;
}

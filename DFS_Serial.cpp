#include <iostream>
#include <vector>
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

    // Explore all adjacent vertices
    for (int neighbor : adj[vertex]) {
        if (!visited[neighbor]) {
            dfsUtil(neighbor, visited);
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
    std::clock_t startTime = std::clock();
    
    std::cout << "DFS starting from vertex 0: ";
    graph.dfs(0);
    std::cout << "\n";

    std::clock_t endTime = std::clock();
    double executionTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC;
    std::cout << "Execution time: " << executionTime << " seconds" << std::endl;

    return 0;
}

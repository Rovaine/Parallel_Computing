#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <ctime>

class Graph {
public:
    Graph(int vertices);
    void addEdge(int v, int w);
    void bfs(int startVertex);

private:
    int vertices; // Number of vertices
    std::vector<std::vector<int>> adj; // Adjacency list
};

Graph::Graph(int vertices) {
    this->vertices = vertices;
    adj.resize(vertices);
}

void Graph::addEdge(int v, int w) {
    adj[v].push_back(w); // Add w to v’s list
}

void Graph::bfs(int startVertex) {
    std::vector<bool> visited(vertices, false);
    std::queue<int> queue;
    
    visited[startVertex] = true;
    queue.push(startVertex);

    while (!queue.empty()) {
        int currentLevelSize = queue.size();
        std::vector<int> nextLevel;

        // Process the current level in parallel
        #pragma omp parallel for
        for (int i = 0; i < currentLevelSize; i++) {
            int vertex = queue.front();
            queue.pop();

            std::cout << vertex << " ";

            for (int neighbor : adj[vertex]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true; // Mark as visited
                    nextLevel.push_back(neighbor); // Collect next level
                }
            }
        }

        // Add the next level to the queue
        for (int neighbor : nextLevel) {
            queue.push(neighbor);
        }
    }
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
    
    std::cout << "BFS starting from vertex 0: ";
    graph.bfs(0);
    std::cout << "\n";

    double endTime = omp_get_wtime();
    std::cout << "Execution time: " << (endTime - startTime) << " seconds" << std::endl;

    return 0;
}

import random, os, logging

DEBUG = logging.DEBUG

N_GRAPH = 6 # num^2 di grafi che verranno generati al raddoppiare dei nodi e degli archi
MIN_NODES = 500 # minimum Number of nodes in the graph
MIN_K = 256 # minimum Number of attachments per node
MIN_PROB = 0.02# minimum Probability of existence for each edge

FIXED_EDGE_NUMBER = 15000

MIN_EDGE_WEIGHT =  1
MAX_EDGE_WEIGHT = 10000

from enum import Enum
# BarabasiAlbertGraph, ErdosRenyiGraph
class GraphTypes(Enum):
    BAG = 1
    ERG = 2

    def Name(self):
        return self.name

class DijkstraAlgoTypes(Enum):
    STATIC = 1
    DYNAMIC = 2
    
    def Name(self):
        return self.name

STATIC_RESULT_FOLDER = os.path.join(os.getcwd(),"Result/Static")
DYNAMIC_RESULT_FOLDER = os.path.join(os.getcwd(),"Result/Dynamic")
RESULT_FILE_TYPE = ".json"

BAGs_FOLDER = os.path.join(os.getcwd(),"BarabasiAlbertGraphs")
ERGs_FOLDER = os.path.join(os.getcwd(),"ErdosRenyiGraphs")

# MAIN.PY SETTING
COMPUTE_FIRST_ALGO_RUN = False
COMPUTE_PROCESS_TIME = False
GRAPH_TO_CHECK = N_GRAPH
# e'  il numero di eventi randomici che avvengono ad ogni esperimento di dijkstra (per ogni grafo)
EVENT_NUMBER_IN_EXP = 25000


class ResultType(Enum):
    Static = 1
    Dynamic = 2

    def Name(self):
        return self.name

# dato un grafo, ritorna una lista di archi (tuple (u,v,w) = (nodo1, nodo2, peso)) di lunghezza edge_number
# oppure tutti gli archi mancanti se edge_number > archi mancanti 
def getMissingEdgeRandomlyFromGraph(graph, edge_number, min_weight, max_weight):
    missingEdges = []
    nodes = graph.numberOfNodes()
    edges = graph.numberOfEdges()
    
    max_missing_edges = (nodes * (nodes - 1) / 2) - edges
    nodeList = [node for node in range(nodes)]

    # finche' non ho costruito una lista con edge_number archi mancanti
    while (len(missingEdges) < edge_number and len(missingEdges) < max_missing_edges):
        # prendo due nodi random dall'insieme dei nodi
        edge = (random.sample(nodeList, 2))
        edge.append(random.randint(min_weight, max_weight))
        # se il grafo non contiene quell'arco E non e' gia' stato aggiunto alla lista degli archi mancanti, lo aggiungo
        if ((graph.hasEdge(edge[0], edge[1]) == False) and missingEdges.__contains__(edge) == False):
            missingEdges.append(edge)

    return missingEdges


# dato un grafo assegno randomicamente un peso compreso tra [min_weight, max_weight] ad ogni arco
def getRandomWeightedGraph(graph, min_weight, max_weight):
    for edge in graph.iterEdges():
        graph.setWeight(edge[0],edge[1],random.randint(min_weight, max_weight))

    return graph


def getInputSetByDoubling(nGraph, minNodes, minK, doubleNodes=True, doubleEdges=False):
    nodesSize = [minNodes]
    kSize = [minK]

    # Doubling
    for x in range(nGraph -1):
        nodesSize.append(nodesSize[-1] * 2)
        kSize.append(kSize[-1] * 2)

    if(doubleEdges==False):
        kSize.clear()
        kSize.append(minK)

    inputSet = [(n,k) for n in nodesSize for k in kSize]
    assert len(inputSet) == len(nodesSize) * len(kSize)
    return inputSet

def clearFolderByType(folder, graphType):
    if(isinstance(graphType, GraphTypes) == False):
        return

    import os

    type = graphType.Name()

    for root, dirs, files in os.walk(folder):
        for file in files:
            if(file.__contains__(type)):   
                os.remove(os.path.join(root, file))
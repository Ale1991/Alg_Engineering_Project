##################################################################
# script utilizzato per generare grafi pesati e non orientati attraverso generatori random
# input BarabasiAlbertGenerator: nGraph, minNodes, minK
# attraverso il doubling experiment, minNodes e minK vengono raddoppiati indipendentemente nGraph volte
# esempio output con nGraph = 3 => 9 BarabasiAlbertGraph:
# (minK\minNodes)     10     20     40
#                 1 (1\10) (1\20) (1\40)
#                 2 (2\10) (2\20) (2\40)
#                 4 (4\10) (4\20) (4\40)
# per Barabasi minK = numero di archi uscenti da un nodo
# BarabasiAlbertGraph File:             BarabasiAlbertGraphs/BAG_x.TabOne
# BarabasiAlbertGraph MissingEdge File: BarabasiAlbertGraphs/missingEdgeForBAG_x.json

##################################################################
# script utilizzato per generare grafi pesati e non orientati attraverso generatori random
# input ErdosRenyiGenerator: nGraph, minNodes, minProb
# attraverso il doubling experiment, minNodes e minProb vengono raddoppiati indipendentemente nGraph volte
# esempio output con nGraph = 3 => 9 ErdosRenyiGraph:
# (minProb\minNodes)         10        20        40
#                    0.02 (0.02\10) (0.02\20) (0.02\40)
#                    0.04 (0.04\10) (0.04\20) (0.04\40)
#                    0.08 (0.08\10) (0.08\20) (0.08\40)
# per Erdos minProb = probabilita' dell' esistenza di ogni arco
# ErdosRenyiGraph File:                 ErdosRenyiGraphs/ERG_x.TabOne
# ErdosRenyiGraph MissingEdge File:     ErdosRenyiGraphs/missingEdgeForERG_x.json
import networkit, pandas, time, sys, logging, utility

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("genGraph")
logger.setLevel(logging.INFO)

N_GRAPH = utility.N_GRAPH # num^2 di grafi che verranno generati al raddoppiare dei nodi e degli archi
MIN_NODES = utility.MIN_NODES # minimum Number of nodes in the graph
MIN_K = utility.MIN_K # minimum Number of attachments per node
MIN_PROB = utility.MIN_PROB # minimum Probability of existence for each edge
FIXED_EDGE_NUMBER = utility.FIXED_EDGE_NUMBER
MIN_EDGE_WEIGHT =  utility.MIN_EDGE_WEIGHT
MAX_EDGE_WEIGHT = utility.MAX_EDGE_WEIGHT


# Generate Graph
def generateGraphByType(nMax, param2, graphType):

    if(isinstance(graphType, utility.GraphTypes) == False):
        return

    if(graphType == utility.GraphTypes.BAG):
        # BarabasiAlbertGenerator(count k, count nMax, count n0 = 0, bool batagelj = true)
        # k = 4 # -> Number of attachments per node
        # nMax = 100 # -> Number of nodes in the graph
        # n0 = 0 # -> FIXED PARAMETER: Number of connected nodes to begin with
        # batagelj = True # -> FIXED PARAMETER: Specifies whether to use Batagelj and Brandes’s method (much faster) rather than the naive one; default: true
        graph = networkit.generators.BarabasiAlbertGenerator(param2, nMax, n0=0, batagelj= True).generate()
    elif(graphType == utility.GraphTypes.ERG):
        # ErdosRenyiGenerator(count nNodes, double prob, directed = False, selfLoops = False)
        # nMax # -> Number of nodes n in the graph. 
        # prob # -> Probability of existence for each edge p.
        # directed = False # -> FIXED PARAMETER: Generates a directed
        # selfLoops = False # -> FIXED PARAMETER: Allows self-loops to be generated (only for directed graphs)
        graph = networkit.generators.ErdosRenyiGenerator(nMax, param2, directed=False, selfLoops=False).generate()

    # converto il grafo non pesato in grafo pesato
    weighted_graph = networkit.graphtools.toWeighted(graph)

    # assegno agli archi di ciascun grafo pesi random compresi fra MIN_EDGE_WEIGHT e MAX_EDGE_WEIGHT
    return utility.getRandomWeightedGraph(weighted_graph, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

# Store Graph
def storeGraphByType(graph, index, graphType):  
    if(isinstance(graphType, utility.GraphTypes) == False):
        return

    start = time.time()
    # salvo i grafi su file
    if(graphType == utility.GraphTypes.BAG):
        networkit.graphio.writeGraph(graph, f"{utility.BAGs_FOLDER}/{graphType.Name()}_{index}.TabOne", networkit.Format.EdgeListTabOne)
    elif(graphType == utility.GraphTypes.ERG):
        networkit.graphio.writeGraph(graph, f"{utility.ERGs_FOLDER}/{graphType.Name()}_{index}.TabOne", networkit.Format.EdgeListTabOne)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{graphType.Name()}_{index}.TabOne saved in {time.time() - start}")

# Compute&Store Graphs missing edges
def storeMissingEdgesByType(graph, index, graphType):
    if(isinstance(graphType, utility.GraphTypes) == False):
        return

    # per ogni grafo, viene calcolata la lista di archi mancanti al grafo
    # il numero di archi e' fissato (FIXED_EDGE_NUMBER)
    # infine, tale lista viene salvata in un file json missingEdgeFor{GraphType}_x.json
    # dove x fa riferimento al grafo precedentemente generato e salvato con nome {GraphType}_x.TabOne
    nodes = graph.numberOfNodes()
    edges = graph.numberOfEdges()
    max_edges =  (nodes * (nodes - 1) / 2) 

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"G=({nodes},{edges}), max_edges: {max_edges} edge_to_compute: {FIXED_EDGE_NUMBER}")

    missingEdges = utility.getMissingEdgeRandomlyFromGraph(graph, FIXED_EDGE_NUMBER, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)
    
    assert edges + len(missingEdges) <= max_edges

    df = pandas.DataFrame(missingEdges, columns =['from_node', 'to_node', 'weight'])

    if(graphType == utility.GraphTypes.BAG):
        pandas.DataFrame.to_json(df, f"{utility.BAGs_FOLDER}/missingEdgeForBAG_{index}.json", indent=4)
    elif(graphType == utility.GraphTypes.ERG):
        pandas.DataFrame.to_json(df, f"{utility.ERGs_FOLDER}/missingEdgeForERG_{index}.json", indent=4)

# Gen, Store and Compute  Graphs missing edges given experiment parameters and graph type
def genGraphByType(graphType):
    if(isinstance(graphType, utility.GraphTypes) == False):
        return

    if(graphType == utility.GraphTypes.BAG):
        inputSet = utility.getInputSetByDoubling(N_GRAPH, MIN_NODES, MIN_K, doubleNodes=True, doubleEdges=False)
    if(graphType == utility.GraphTypes.ERG):
        inputSet = utility.getInputSetByDoubling(N_GRAPH, MIN_NODES, MIN_PROB, doubleNodes=True, doubleEdges=False)

    for input in inputSet:
        nMax, param2 = input
        index = inputSet.index(input)+1
        graph = generateGraphByType(nMax, param2, graphType) # genero l'i-esimo grafo
        storeGraphByType(graph, index, graphType) # salvo l'i-esimo grafo
        storeMissingEdgesByType(graph, index, graphType) # calcolo e salvo gli archi mancanti dell'i-esimo grafo

if __name__ == "__main__":
    utility.clearFolderByType(utility.BAGs_FOLDER, utility.GraphTypes.BAG)
    genGraphByType(utility.GraphTypes.BAG)

    # genGraphByType(utility.GraphTypes.ERG)

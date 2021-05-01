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
import networkit, pandas, time, sys, logging
import utility

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("genBAGraph")
logger.setLevel(logging.DEBUG)

N_GRAPH = utility.N_GRAPH # num^2 di grafi che verranno generati al raddoppiare dei nodi e degli archi
MIN_NODES = utility.MIN_NODES # minimum Number of nodes in the graph
MIN_K = utility.MIN_K # minimum Number of attachments per node
FIXED_EDGE_NUMBER = utility.FIXED_EDGE_NUMBER
MIN_EDGE_WEIGHT =  utility.MIN_EDGE_WEIGHT
MAX_EDGE_WEIGHT = utility.MAX_EDGE_WEIGHT

# BAG: BarabasiAlbertGraph
def saveSingleBAG_MissingEdges(counter, graph):
    # per ogni grafo BarabasiAlbert, viene calcolata la lista di archi mancanti al grafo
    # il numero di archi e' pari ad una percentuale del numero di archi del corrispondente grafo completo
    # infine, tale lista viene salvata in un file json missingEdgeForBAG_x.json
    # dove x fa riferimento al grafo precedentemente generato e salvato con nome BAG_x.TabOne
    nodes = graph.numberOfNodes()
    edges = graph.numberOfEdges()
    max_edges =  (nodes * (nodes - 1) / 2) 

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"G=({nodes},{edges}), max_edges: {max_edges} edge_to_compute: {FIXED_EDGE_NUMBER}")
    missingEdges = utility.getMissingEdgeRandomlyFromGraph(graph, FIXED_EDGE_NUMBER, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)
    
    assert edges + len(missingEdges) <= max_edges
    df = pandas.DataFrame(missingEdges, columns =['from_node', 'to_node', 'weight'])
    pandas.DataFrame.to_json(df, f"BarabasiAlbertGraphs/missingEdgeForBAG_{counter}.json", indent=4)

# genero grafo, salvo grafo, calcolo missingEdge e salvo missingEdge
def genAndStoreSingleBAG(nMax, k, index):
    # BarabasiAlbertGenerator(count k, count nMax, count n0 = 0, bool batagelj = true)
    # k = 4 # -> Number of attachments per node
    # nMax = 100 # -> Number of nodes in the graph
    n0 = 0 # -> FIXED PARAMETER: Number of connected nodes to begin with
    batagelj = True # -> FIXED PARAMETER: Specifies whether to use Batagelj and Brandes’s method (much faster) rather than the naive one; default: true

    start = time.time()
    bag = networkit.generators.BarabasiAlbertGenerator(k, nMax, n0, batagelj).generate()

    # converto il grafo non pesato in grafo pesato
    weighted_bag = networkit.graphtools.toWeighted(bag)

    # assegno agli archi di ciascun grafo pesi random compresi fra MIN_EDGE_WEIGHT e MAX_EDGE_WEIGHT
    random_weighted_bag = utility.getRandomWeightedGraph(weighted_bag, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)
  
    # salvo i grafi su file
    networkit.graphio.writeGraph(random_weighted_bag, f"BarabasiAlbertGraphs/BAG_{index}.TabOne", networkit.Format.EdgeListTabOne)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"BAG_{index}.TabOne saved in {time.time() - start}")

    start_missing = time.time()
    saveSingleBAG_MissingEdges(index, random_weighted_bag)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"missingEdge_ {index} saved in {time.time() - start_missing}")

def genDefaultERG():
    inputSet = utility.getInputSetByDoubling(N_GRAPH, MIN_NODES, MIN_K, doubleNodes=True, doubleEdges=False)
    for input in inputSet:
        nMax, k = input
        genAndStoreSingleBAG(nMax, k, inputSet.index(input)+1)


if __name__ == "__main__":
    genDefaultERG()

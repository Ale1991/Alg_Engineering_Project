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
logger = logging.getLogger("genERGraph")
logger.setLevel(logging.DEBUG)

N_GRAPH = utility.N_GRAPH # num^2 di grafi che verranno generati al raddoppiare dei nodi e degli archi
MIN_NODES = utility.MIN_NODES # minimum Number of nodes in the graph
MIN_PROB = utility.MIN_PROB # minimum Probability of existence for each edge
FIXED_EDGE_NUMBER = utility.FIXED_EDGE_NUMBER
MIN_EDGE_WEIGHT =  utility.MIN_EDGE_WEIGHT
MAX_EDGE_WEIGHT = utility.MAX_EDGE_WEIGHT

# BAG: ErdosRenyiGraph
def saveSingleERG_MissingEdges(counter, graph):
    # per ogni grafo ErdosRenyi, viene calcolata la lista di archi mancanti al grafo
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
    pandas.DataFrame.to_json(df, f"ErdosRenyiGraphs/missingEdgeForERG_{counter}.json", indent=4)

# genero grafo, salvo grafo, calcolo missingEdge e salvo missingEdge
def genAndStoreSingleErdosRenyiGraph(nMax, prob, index):  
    # ErdosRenyiGenerator(count nNodes, double prob, directed = False, selfLoops = False)
    # nMax # -> Number of nodes n in the graph. 
    # prob # -> Probability of existence for each edge p.
    directed = False # -> FIXED PARAMETER: Generates a directed
    selfLoops = False # -> FIXED PARAMETER: Allows self-loops to be generated (only for directed graphs)

    start = time.time()
    erg = networkit.generators.ErdosRenyiGenerator(nMax, prob, directed, selfLoops).generate()

    weighted_erg = networkit.graphtools.toWeighted(erg)
    # assegno agli archi di ciascun grafo pesi random compresi fra MIN_EDGE_WEIGHT e MAX_EDGE_WEIGHT
    random_weighted_erg = utility.getRandomWeightedGraph(weighted_erg, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

    # salvo i grafi su file
    networkit.graphio.writeGraph(random_weighted_erg, f"ErdosRenyiGraphs/ERG_{index}.TabOne", networkit.Format.EdgeListTabOne)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"ERG_{index}.TabOne saved in {time.time() - start}")

    start_missing = time.time()
    saveSingleERG_MissingEdges(index, random_weighted_erg)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"missingEdge_{index} saved in {time.time() - start_missing}")

def genDefaultBAG():
    inputSet = utility.getInputSetByDoubling(N_GRAPH, MIN_NODES, MIN_PROB, doubleNodes=True, doubleEdges=False)
    for input in inputSet:
        nMax, prob = input
        genAndStoreSingleErdosRenyiGraph(nMax, prob, inputSet.index(input)+1)


if __name__ == "__main__":
    genDefaultBAG()

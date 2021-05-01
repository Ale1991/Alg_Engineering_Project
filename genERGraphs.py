##################################################################
# script utilizzato per generare grafi pesati e non orientati attraverso generatori random
# input BarabasiAlbertGenerator: nGraph, minNodes, minK
# attraverso il doubling experiment, minNodes e minK vengono raddoppiati indipendentemente nGraph volte
# esempio output con nGraph = 3 => 9 grafi Barabasi e 9 Erdos:
# minK\minNodes    10   20    40
#              1 1\10 1\20  1\40
#              2 2\10 2\20  2\40
#              4 4\10 4\20  4\40
# per Barabasi minK = numero di archi uscenti da un nodo
# per Erdos minProb = probabilita' dell' esistenza di ogni arco
# ErdosRenyiGraph File:                 ErdosRenyiGraphs/ERG_x.TabOne
# ErdosRenyiGraph MissingEdge File:     ErdosRenyiGraphs/missingEdgeForERG_x.json
import random, networkit, pandas, time
import genBAGraph

N_GRAPH = genBAGraph.N_GRAPH # num^2 di grafi che verranno generati al raddoppiare dei nodi e degli archi
MIN_NODES = genBAGraph.MIN_NODES # minimum Number of nodes in the graph
MIN_PROB = 0.02# minimum Probability of existence for each edge

FIXED_EDGE_NUMBER = genBAGraph.FIXED_EDGE_NUMBER

MIN_EDGE_WEIGHT =  genBAGraph.MIN_EDGE_WEIGHT
MAX_EDGE_WEIGHT = genBAGraph.MAX_EDGE_WEIGHT

# dato un grafo, ritorna una lista di archi (tuple (u,v,w) = (nodo1, nodo2, peso)) di lunghezza edge_number
# oppure tutti gli archi mancanti se edge_number > archi mancanti 
def getMissingEdgeRandomlyFromGraph(graph, edge_number):
    missingEdges = []
    nodes = graph.numberOfNodes()
    edges = graph.numberOfEdges()
    
    max_missing_edges = (nodes * (nodes - 1) / 2) - edges

    nodeList = [node for node in range(nodes)]

    # finche' non ho costruito una lista con edge_number archi mancanti
    while (len(missingEdges) < edge_number and len(missingEdges) < max_missing_edges):
        # prendo due nodi random dall'insieme dei nodi
        edge = (random.sample(nodeList, 2))
        edge.append(random.randint(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT))
        # se il grafo non contiene quell'arco E non e' gia' stato aggiunto alla lista degli archi mancanti, lo aggiungo
        if ((graph.hasEdge(edge[0], edge[1]) == False) and missingEdges.__contains__(edge) == False):
            missingEdges.append(edge)

    # print(f"archi da trovare: {edge_number}, archi trovati: {len(missingEdges)}")
    # print(f'#node:{nodes}, #edge:{edges}, #theoreticallyMissingEdges:{max_missing_edges}, #missingEdges added:{len(missingEdges)}')

    return missingEdges

# BAG: ErdosRenyiGraph
def saveSingleERG_MissingEdges(counter, graph):
    # per ogni grafo ErdosRenyi, viene calcolata la lista di archi mancanti al grafo
    # il numero di archi e' pari ad una percentuale del numero di archi del corrispondente grafo completo
    # infine, tale lista viene salvata in un file json missingEdgeForBAG_x.json
    # dove x fa riferimento al grafo precedentemente generato e salvato con nome BAG_x.TabOne
    nodes = graph.numberOfNodes()
    edges = graph.numberOfEdges()
    max_edges =  (nodes * (nodes - 1) / 2) 

    print(f"G=({nodes},{edges}), max_edges: {max_edges} edge_to_compute: {FIXED_EDGE_NUMBER}")
    missingEdges = getMissingEdgeRandomlyFromGraph(graph, FIXED_EDGE_NUMBER)
    
    assert edges + len(missingEdges) <= max_edges
    df = pandas.DataFrame(missingEdges, columns =['from_node', 'to_node', 'weight'])
    pandas.DataFrame.to_json(df, f"ErdosRenyiGraphs/missingEdgeForERG_{counter}.json", indent=4)

# genero grafo, salvo grafo, calcolo missingEdge e salvo missingEdge
# genero nGraph ErdosRenyi raddoppiando ogni volta minNodes e minProb (doubling)
def genAndStoreErdosRenyiGraph(nGraph, minNodes, minProb):
    nodesSize = [minNodes]
    probSize = [minProb]

    for x in range(nGraph -1):
        nodesSize.append(nodesSize[-1] * 2)

        p = probSize[-1] * 2
        if(p >= 1):
            p = 0.98
        probSize.append(p)

    # inputSet = [(n,k) for n in nodesSize for k in probSize]

    probSize.clear()
    probSize.append(minProb)
    inputSet = [(n,k) for n in nodesSize for k in probSize]

    assert len(inputSet) == len(nodesSize) * len(probSize)
    
    # Bases: networkit.generators.StaticGraphGenerator
    # The generation follows Vladimir Batagelj and Ulrik Brandes: "Efficient generation of large random networks", Phys Rev E 71, 036113 (2005).
    # ErdosRenyiGenerator(count nNodes, double prob, directed = False, selfLoops = False)
    # Creates G(nNodes, prob) graphs.
    # nNodes : count    Number of nodes n in the graph.
    # nNodes = 100 
    # prob : double     Probability of existence for each edge p.
    # prob = 0.2
    # directed : bool   Generates a directed
    directed = False # -> FIXED PARAMETER
    # selfLoops : bool  Allows self-loops to be generated (only for directed graphs)
    selfLoops = False # -> FIXED PARAMETER

    counter = 0
    for t in inputSet:
        counter +=1
        nNodes, prob = t
        start = time.time()
        erg = networkit.generators.ErdosRenyiGenerator(nNodes, prob, directed, selfLoops).generate()

        weighted_erg = networkit.graphtools.toWeighted(erg)
        # assegno agli archi di ciascun grafo pesi random compresi fra MIN_EDGE_WEIGHT e MAX_EDGE_WEIGHT
        random_weighted_erg = genBAGraph.getRandomWeightedGraph(weighted_erg, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)

        # salvo i grafi su file
        networkit.graphio.writeGraph(random_weighted_erg, f"ErdosRenyiGraphs/ERG_{counter}.TabOne", networkit.Format.EdgeListTabOne)
        print(f"ERG_{counter}.TabOne saved in {time.time() - start}")

        start_missing = time.time()
        saveSingleERG_MissingEdges(counter, random_weighted_erg)
        print(f"missingEdge_{counter} saved in {time.time() - start_missing}")


if __name__ == "__main__":
    genAndStoreErdosRenyiGraph(N_GRAPH, MIN_NODES, MIN_PROB)
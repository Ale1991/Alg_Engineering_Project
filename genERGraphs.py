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

PERCENTAGE_OF_MISSING_EDGE_TO_COMPUTE = genBAGraph.PERCENTAGE_OF_MISSING_EDGE_TO_COMPUTE


FIXED_EDGE_NUMBER = genBAGraph.FIXED_EDGE_NUMBER
FIXED_EDGE = genBAGraph.FIXED_EDGE

MIN_EDGE_WEIGHT =  genBAGraph.MIN_EDGE_WEIGHT
MAX_EDGE_WEIGHT = genBAGraph.MAX_EDGE_WEIGHT

GEN_BAG = False
GEN_NEW_BAG = True
GEN_ERG = False

def getRandomWeightedGraph(graph):
    # counter = 0
    # for graph in graphs:
    # start_rndWeight = time.time()
    for edge in graph.iterEdges():
        # print(f"weight_before: {graph.weight(edge[0], edge[1])}")
        graph.setWeight(edge[0],edge[1],random.randint(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT))
        # print(f"weight_after: {graph.weight(edge[0], edge[1])}")
    # counter+=1
        # print(f"graph{counter} weight assigned in {time.time()-start_rndWeight}")
    return graph

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

    if(FIXED_EDGE):
        edge_to_compute = FIXED_EDGE_NUMBER
    else:
        edge_to_compute = int(max_edges * PERCENTAGE_OF_MISSING_EDGE_TO_COMPUTE)

    print(f"G=({nodes},{edges}), max_edges: {max_edges} edge_to_compute: {edge_to_compute}")
    missingEdges = getMissingEdgeRandomlyFromGraph(graph, edge_to_compute)
    
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
        random_weighted_erg = getRandomWeightedGraph(weighted_erg)

        # salvo i grafi su file
        networkit.graphio.writeGraph(random_weighted_erg, f"ErdosRenyiGraphs/ERG_{counter}.TabOne", networkit.Format.EdgeListTabOne)
        print(f"ERG_{counter}.TabOne saved in {time.time() - start}")

        start_missing = time.time()
        saveSingleERG_MissingEdges(counter, random_weighted_erg)
        print(f"missingEdge_{counter} saved in {time.time() - start_missing}")


if __name__ == "__main__":

    if(GEN_NEW_BAG):
        genAndStoreErdosRenyiGraph(N_GRAPH, MIN_NODES, MIN_PROB)


    # -> PREPROCESSING: inutile ricalcolare i grafi ogni volta, vogliamo analizzare lo speedup dei due algo di dijkstra
    # al variare delle caratteristiche dei grafi
    # TODO generare un metodo random per gli eventi dinamici
    # event = networkit.dynamic.GraphEvent(networkit.dynamic.GraphEvent.EDGE_ADDITION, missing_edge[0],missing_edge[1], 0)


# cambio radice sssp per ammortizzare il bias (10 di cambi)
# 4 exp per cambio taglia nodi, 4 per archi 
# 4 cambi di tipologia di grafi (barabasi, erdos, qualche grafo reale)
# grafici con running time (ordinate) ascisse ( vertici,archi, densita)



# NOT USED FUNC
# ERG: ErdosRenyiGraph
def saveERG_MissingEdges(erg_list):
    # per ogni grafo ErdosRenyi, viene calcolata la lista di archi mancanti al grafo
    # il numero di archi e' pari ad una percentuale del numero di archi del corrispondente grafo completo
    # infine, tale lista viene salvata in un file json missingEdgeForERG_x.json
    # dove x fa riferimento al grafo precedentemente generato e salvato con nome ERG_x.TabOne
    counter = 0
    for graph in erg_list:
        counter += 1
        
        nodes = graph.numberOfNodes()
        edges = graph.numberOfEdges()


        max_edges =  (nodes * (nodes - 1) / 2) 

        if(FIXED_EDGE):
            edge_to_compute = FIXED_EDGE_NUMBER
        else:
            edge_to_compute = int(max_edges * PERCENTAGE_OF_MISSING_EDGE_TO_COMPUTE)
            
        missingEdges = getMissingEdgeRandomlyFromGraph(graph, edge_to_compute)
        
        assert edges + len(missingEdges) <= max_edges

        df = pandas.DataFrame(missingEdges, columns = ['from_node', 'to_node', 'weight'])
        pandas.DataFrame.to_json(df, f"ErdosRenyiGraphs/missingEdgeForERG_{counter}.json", indent=4)


# genero nGraph ErdosRenyi raddoppiando ogni volta minNodes e minProb (doubling)
def genErdosRenyiGraph(nGraph, minNodes, minProb):
    nodesSize = [minNodes]
    probSize = [minProb]

    for x in range(nGraph -1):
        nodesSize.append(nodesSize[-1] * 2)

        p = probSize[-1] * 2
        if(p > 1):
            p = 1
        probSize.append(p)

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

    erdos_renyi_graphs = []
    for t in inputSet:
        nNodes, prob = t
        # print(f"tupla: {t}, nNodes:{nNodes}, prob:{prob}")
        erdos_renyi_graphs.append(networkit.generators.ErdosRenyiGenerator(nNodes, prob, directed, selfLoops).generate())

    return erdos_renyi_graphs

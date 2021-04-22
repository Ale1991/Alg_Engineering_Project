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
# BarabasiAlbertGraph File:             BarabasiAlbertGraphs/BAG_x.TabOne
# BarabasiAlbertGraph MissingEdge File: BarabasiAlbertGraphs/missingEdgeForBAG_x.json
# ErdosRenyiGraph File:                 ErdosRenyiGraphs/ERG_x.TabOne
# ErdosRenyiGraph MissingEdge File:     ErdosRenyiGraphs/missingEdgeForERG_x.json
import random, networkit, pandas, time

N_GRAPH = 6 # num^2 di grafi che verranno generati al raddoppiare dei nodi e degli archi
MIN_NODES = 500 # minimum Number of nodes in the graph
MIN_K = 4 # minimum Number of attachments per node

PERCENTAGE_OF_MISSING_EDGE_TO_COMPUTE = 0.00001

FIXED_EDGE_NUMBER = 5000
FIXED_EDGE = True

MIN_EDGE_WEIGHT =  10
MAX_EDGE_WEIGHT = 20

GEN_BAG = False
GEN_NEW_BAG = True
GEN_ERG = False

# non e' fattibile per grafi con piu di 700 nodi
def getMissingEdgeFromGraph(graph):
    missingEdges = []
    for u in graph.iterNodes():
        for v in graph.iterNodes():
            if (u != v and graph.hasEdge(u, v) == False and missingEdges.__contains__((v, u)) == False):
                missingEdges.append((u, v))

    # print(missingEdges)
    n = graph.numberOfNodes()
    e = graph.numberOfEdges()
    # print(f'#node:{n}, #edge:{e}, #theoreticallyMissingEdges:{(n * (n - 1) / 2) - e}, #missingEdges:{len(missingEdges)}')
    return missingEdges

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

# genero nGraph BarabasiAlbert raddoppiando ogni volta minNodes e minK (doubling)
def genBarabasiAlbertGraph(nGraph, minNodes, minK):
    nodesSize = [minNodes]
    kSize = [minK]

    for x in range(nGraph -1):
        nodesSize.append(nodesSize[-1] * 2)
        kSize.append(kSize[-1] * 2)

    inputSet = [(n,k) for n in nodesSize for k in kSize]

    # assert len(inputSet) == len(nodesSize) * len(kSize)
    
    # BarabasiAlbertGenerator(count k, count nMax, count n0 = 0, bool batagelj = true)
    # k: Number of attachments per node
    # k = 4
    # nMax: Number of nodes in the graph
    # nMax = 100
    # n0: Number of connected nodes to begin with
    n0 = 0 # -> FIXED PARAMETER
    # batagelj: Specifies whether to use Batagelj and Brandes’s method (much faster) rather than the naive one; default: true
    batagelj = True # -> FIXED PARAMETER

    barabasi_albert_graphs = []
    for t in inputSet:
        nMax, k = t
        barabasi_albert_graphs.append(networkit.generators.BarabasiAlbertGenerator(k, nMax, n0, batagelj).generate())

    return barabasi_albert_graphs


def getRandomWeightedGraphs(graphs):
    counter = 0
    for graph in graphs:
        start_rndWeight = time.time()
        for edge in graph.iterEdges():
            # print(f"weight_before: {graph.weight(edge[0], edge[1])}")
            graph.setWeight(edge[0],edge[1],random.randint(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT))
            # print(f"weight_after: {graph.weight(edge[0], edge[1])}")
        counter+=1
        print(f"graph{counter} weight assigned in {time.time()-start_rndWeight}")
    return graphs

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



# BAG: BarabasiAlbertGraph
def saveBAG_MissingEdges(bag_list):
    # per ogni grafo BarabasiAlbert, viene calcolata la lista di archi mancanti al grafo
    # il numero di archi e' pari ad una percentuale del numero di archi del corrispondente grafo completo
    # infine, tale lista viene salvata in un file json missingEdgeForBAG_x.json
    # dove x fa riferimento al grafo precedentemente generato e salvato con nome BAG_x.TabOne
    counter = 0
    for graph in bag_list:
        counter += 1

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
        pandas.DataFrame.to_json(df, f"BarabasiAlbertGraphs/missingEdgeForBAG_{counter}.json", indent=4)


# BAG: BarabasiAlbertGraph
def saveSingleBAG_MissingEdges(counter, graph):
    # per ogni grafo BarabasiAlbert, viene calcolata la lista di archi mancanti al grafo
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
    # missingEdges = getMissingEdgeRandomlyFromGraph(graph, max_edges * percentage_of_missing_edge_to_compute)
    
    assert edges + len(missingEdges) <= max_edges
    df = pandas.DataFrame(missingEdges, columns =['from_node', 'to_node', 'weight'])
    pandas.DataFrame.to_json(df, f"BarabasiAlbertGraphs/missingEdgeForBAG_{counter}.json", indent=4)

# genero grafo, salvo grafo, calcolo missingEdge e salvo missingEdge
def genAndStoreBarabasiAlbertGraph(nGraph, minNodes, minK):
    nodesSize = [minNodes]
    kSize = [minK]

    for x in range(nGraph -1):
        nodesSize.append(nodesSize[-1] * 2)
        kSize.append(kSize[-1] * 2)

    kSize.clear()
    kSize.append(minK)
    inputSet = [(n,k) for n in nodesSize for k in kSize]

    # assert len(inputSet) == len(nodesSize) * len(kSize)
    
    # BarabasiAlbertGenerator(count k, count nMax, count n0 = 0, bool batagelj = true)
    # k: Number of attachments per node
    # k = 4
    # nMax: Number of nodes in the graph
    # nMax = 100
    # n0: Number of connected nodes to begin with
    n0 = 0 # -> FIXED PARAMETER
    # batagelj: Specifies whether to use Batagelj and Brandes’s method (much faster) rather than the naive one; default: true
    batagelj = True # -> FIXED PARAMETER

    counter = 0
    for t in inputSet:
        counter +=1
        start = time.time()
        nMax, k = t
        bag = networkit.generators.BarabasiAlbertGenerator(k, nMax, n0, batagelj).generate()

        weighted_bag = networkit.graphtools.toWeighted(bag)
        # assegno agli archi di ciascun grafo pesi random compresi fra MIN_EDGE_WEIGHT e MAX_EDGE_WEIGHT
        random_weighted_bag = getRandomWeightedGraph(weighted_bag)
        

        # salvo i grafi su file
        networkit.graphio.writeGraph(random_weighted_bag, f"BarabasiAlbertGraphs/BAG_{counter}.TabOne", networkit.Format.EdgeListTabOne)
        print(f"BAG_{counter}.TabOne saved in {time.time() - start}")

        start_missing = time.time()
        saveSingleBAG_MissingEdges(counter, random_weighted_bag)
        print(f"missingEdge_{counter} saved in {time.time() - start_missing}")


if __name__ == "__main__":
    if(GEN_BAG):
        # genero i grafi secondo Barabasi Albert
        barabasi_graphs = genBarabasiAlbertGraph(N_GRAPH, MIN_NODES, MIN_K)
        # converto tali grafi in grafi pesati
        weighted_barabasi_graphs = [networkit.graphtools.toWeighted(x) for x in barabasi_graphs]
        # assegno agli archi di ciascun grafo pesi random compresi fra MIN_EDGE_WEIGHT e MAX_EDGE_WEIGHT
        weighted_barabasi_graphs = getRandomWeightedGraphs(weighted_barabasi_graphs)
        counter = 0

        # salvo i grafi su file
        for graph in weighted_barabasi_graphs:
            counter +=1
            networkit.graphio.writeGraph(graph, f"BarabasiAlbertGraphs/BAG_{counter}.TabOne", networkit.Format.EdgeListTabOne)

        saveBAG_MissingEdges(weighted_barabasi_graphs)

    if(GEN_NEW_BAG):
        genAndStoreBarabasiAlbertGraph(N_GRAPH, MIN_NODES, MIN_K)


    # -> PREPROCESSING: inutile ricalcolare i grafi ogni volta, vogliamo analizzare lo speedup dei due algo di dijkstra
    # al variare delle caratteristiche dei grafi
    # TODO generare un metodo random per gli eventi dinamici
    # event = networkit.dynamic.GraphEvent(networkit.dynamic.GraphEvent.EDGE_ADDITION, missing_edge[0],missing_edge[1], 0)


# cambio radice sssp per ammortizzare il bias (10 di cambi)
# 4 exp per cambio taglia nodi, 4 per archi 
# 4 cambi di tipologia di grafi (barabasi, erdos, qualche grafo reale)
# grafici con running time (ordinate) ascisse ( vertici,archi, densita)

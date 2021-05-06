from utility import ERGs_FOLDER, GraphTypes, MAX_EDGE_WEIGHT, MIN_EDGE_WEIGHT
import logging, time, numpy, random, networkit, graphParser, copy, sys, timeit, pandas, utility

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("testDijkstra")
logger.setLevel(logging.INFO)


# return random(networkit.dynamic.GraphEvent.EDGE_ADDITION, networkit.dynamic.GraphEvent.EDGE_WEIGHT_UPDATE)
def getRandomGraphEdgeEvent():
    # All possible graph event for DynDijkstra
    # graph_event_list = [networkit.dynamic.GraphEvent.EDGE_ADDITION,networkit.dynamic.GraphEvent.EDGE_REMOVAL,
    #                    networkit.dynamic.GraphEvent.EDGE_WEIGHT_INCREMENT,networkit.dynamic.GraphEvent.EDGE_WEIGHT_UPDATE]

    # DynDijkstra Graph cannot be updated with removal and increment event
    # graph_event_list = [networkit.dynamic.GraphEvent.EDGE_REMOVAL, networkit.dynamic.GraphEvent.EDGE_WEIGHT_INCREMENT]

    # Correct graph event for DynDijkstra
    graph_event_list = [networkit.dynamic.GraphEvent.EDGE_ADDITION, networkit.dynamic.GraphEvent.EDGE_WEIGHT_UPDATE]

    return random.choice(graph_event_list)

# return CPU/Process time
def computeDijkstra(sssp):
    if(utility.COMPUTE_PROCESS_TIME == True):
        start_time = timeit.default_timer()# Process time
        sssp.run()
        return (timeit.default_timer() - start_time)
    elif(utility.COMPUTE_PROCESS_TIME == False):
        start_time = time.process_time()# CPU time
        sssp.run()
        return (time.process_time() - start_time)
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("logic error")

# return CPU/Process time
def computeDynDijkstra(dynDijkstra, event):
    if(utility.COMPUTE_PROCESS_TIME == True):
        if(type(event) is networkit.dynamic.GraphEvent):
            start_time = timeit.default_timer()# Process time
            dynDijkstra.update(event)
            return (timeit.default_timer() - start_time)
        else:# FIRST_RUN
            start_time = timeit.default_timer()# Process time
            dynDijkstra.run()
            return (timeit.default_timer() - start_time)
    elif(utility.COMPUTE_PROCESS_TIME == False):  
        if(type(event) is networkit.dynamic.GraphEvent):
            start_time = time.process_time() # CPU time
            dynDijkstra.update(event)
            return (time.process_time() - start_time)
        else:# FIRST_RUN
            start_time = time.process_time() # CPU time
            dynDijkstra.run()
            return (time.process_time() - start_time)
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("logic error")

def DijkstraWithRandomEventTest(graph, event_number, missing_edge_to_add):
    # Dijkstra’s SSSP algorithm
    static_computing_time_list = []
    # new local graph instance foreach experiment
    localGraph = copy.copy(graph)

    # DynDijkstra’s SSSP algorithm
    dynamic_computing_time_list = []
    # new local graph instance foreach experiment
    dyn_localGraph = copy.copy(graph)

    # first dijkstra computation without adding missing edge
    sssp = networkit.distance.Dijkstra(localGraph, 0)

    # first DynDijkstra computation without adding missing edge
    dynSssp = networkit.distance.DynDijkstra(dyn_localGraph, 0)

    # tupla con (evento, #nodi, #archi, tempo)
    if(utility.COMPUTE_FIRST_ALGO_RUN):
        static_computing_time_list.append(("FIRST_RUN", computeDijkstra(sssp)))
        dynamic_computing_time_list.append(("FIRST_RUN", computeDynDijkstra(dynSssp, "FIRST_RUN")))
    else:
        sssp.run()
        dynSssp.run()

    if logger.isEnabledFor(logging.DEBUG):
        assert dynSssp.getDistances() == sssp.getDistances()

    # parte da 1 perche missing_edge_to_add in posizione 0 contiene le stringhe header
    edge_addition_counter = 1
    
    for i in range(event_number):
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"edge_addition_counter: {edge_addition_counter}")

        # scelgo randomicamente uno dei 4 eventi tra EDGE_ADDITION, EDGE_REMOVAL, EDGE_WEIGHT_INCREMENT, EDGE_WEIGHT_UPDATE
        event = getRandomGraphEdgeEvent()

        # scelgo randomicamente un nodo sorgente da cui calcolare sssp per ammortizzare il bias
        random_source_node = 0 # networkit.graphtools.randomNode(localGraph)

        sssp.setSource(random_source_node)
        dynSssp.setSource(random_source_node)

        if(event == networkit.dynamic.GraphEvent.EDGE_ADDITION):
            edge_addition_counter = handleEdgeAdditionEvent(event, localGraph, dyn_localGraph, sssp, dynSssp, static_computing_time_list, dynamic_computing_time_list,
            edge_addition_counter, missing_edge_to_add)
        elif(event == networkit.dynamic.GraphEvent.EDGE_REMOVAL):
            handleEdgeRemovalEvent(event, localGraph, dyn_localGraph, sssp, dynSssp, static_computing_time_list, dynamic_computing_time_list)
        elif(event == networkit.dynamic.GraphEvent.EDGE_WEIGHT_INCREMENT):
            handleEdgeWeightIncrementEvent(event, localGraph, dyn_localGraph, sssp, dynSssp, static_computing_time_list, dynamic_computing_time_list)
        elif(event == networkit.dynamic.GraphEvent.EDGE_WEIGHT_UPDATE):
            handleEdgeWeightUpdateEvent(event, localGraph, dyn_localGraph, sssp, dynSssp, static_computing_time_list, dynamic_computing_time_list)
        
    return [static_computing_time_list, dynamic_computing_time_list]

# calcola ed inserisce il RunningTime impiegato da Dijkstra e DynDijkstra per calcolare SSSP a fronte di un
# evento EDGE_REMOVAL
def handleEdgeAdditionEvent(event, localGraph, dyn_localGraph, sssp, dynSssp, static_computing_time_list, dynamic_computing_time_list, edge_addition_counter, missing_edge_to_add):
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.numberOfNodes() == dyn_localGraph.numberOfNodes()
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.numberOfEdges() == dyn_localGraph.numberOfEdges()

    # COMMENTED PER EVITARE DI PRENDERE ARCHI MANCANTI DAI FILE JSON PRECALCOLATI
    # SOSTITUITO CON GENERAZIONE RANDOM IN LOCO
    # controllo che gli archi aggiunti siano minori della taglia dei missingEdges calcolati e salvati nel json
    if logger.isEnabledFor(logging.DEBUG):
        assert edge_addition_counter < missing_edge_to_add.index.size

    from_node = missing_edge_to_add['from_node'][edge_addition_counter]
    to_node = missing_edge_to_add['to_node'][edge_addition_counter]
    weight = missing_edge_to_add['weight'][edge_addition_counter]

    # controllo che i nodi dell'arco da aggiungere siano diversi
    if logger.isEnabledFor(logging.DEBUG):
        assert from_node != to_node

    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.hasEdge(from_node,to_node) == dyn_localGraph.hasEdge(from_node, to_node)
            
    # EDGE ADDITION EVENT
    localGraph.addEdge(from_node, to_node, weight)
    dyn_localGraph.addEdge(from_node, to_node, weight)

    edge_addition_counter+=1
    static_computing_time_list.append(("EDGE_ADDITION", computeDijkstra(sssp)))
            
    dyn_event = networkit.dynamic.GraphEvent(event, from_node, to_node, weight)
    dynamic_computing_time_list.append(("EDGE_ADDITION", computeDynDijkstra(dynSssp, dyn_event)))

    # controllo che l'arco sia stato aggiunto in entrambi i grafi
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.hasEdge(from_node,to_node) == dyn_localGraph.hasEdge(from_node, to_node)
        # controllo che il peso dell'arco aggiunto sia uguale in entrambi i grafi
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.weight(from_node, to_node) == dyn_localGraph.weight(from_node, to_node)

    # controllo che i cammini minimi siano uguali 
    if logger.isEnabledFor(logging.DEBUG):
        assert dynSssp.distance(to_node) == sssp.distance(to_node)

    # # se i path son diversi e' sufficiente che abbiano la stessa distanza
    #      if logger.isEnabledFor(logging.DEBUG):
    # assert dynSssp.getPath(to_node)  == sssp.getPath(to_node)

    # controllo che il peso totale dei due grafi sia uguale
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.totalEdgeWeight() == dyn_localGraph.totalEdgeWeight()

    return edge_addition_counter

# calcola ed inserisce il RunningTime impiegato da Dijkstra e DynDijkstra per calcolare SSSP a fronte di un
# evento EDGE_REMOVAL
def handleEdgeRemovalEvent(event, localGraph, dyn_localGraph, sssp, dynSssp, static_computing_time_list, dynamic_computing_time_list):
    # randomEdge(G, uniformDistribution=False) returns a random edge of graph G. 
    # If uniformDistribution is set to True, the edge is selected uniformly at random.
    from_node, to_node = networkit.graphtools.randomEdge(localGraph, True)
    new_weight = MAX_EDGE_WEIGHT

    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.hasEdge(from_node, to_node) == True
    if logger.isEnabledFor(logging.DEBUG):
        assert dyn_localGraph.hasEdge(from_node, to_node) == True

    localGraph.setWeight(from_node, to_node, new_weight)
    dyn_localGraph.setWeight(from_node, to_node, new_weight)
    static_computing_time_list.append(("EDGE_REMOVAL", computeDijkstra(sssp)))
    # simulo la rimozione di un arco assegnandogli peso massimo attraverso l'evento EDGE_WEIGHT_UPDATE 
    # poiche' DynDijkstra non supporta l' evento EDGE_REMOVAL
    dyn_event = networkit.dynamic.GraphEvent(networkit.dynamic.GraphEvent.EDGE_WEIGHT_UPDATE, from_node, to_node, new_weight)
    dynamic_computing_time_list.append(("EDGE_REMOVAL", computeDynDijkstra(dynSssp, dyn_event)))
    
    # controllo che l'arco sia stato aggiunto in entrambi i grafi
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.hasEdge(from_node,to_node) == dyn_localGraph.hasEdge(from_node, to_node)
    # controllo che il peso dell'arco aggiunto sia uguale in entrambi i grafi
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.weight(from_node, to_node) == dyn_localGraph.weight(from_node, to_node)
    # controllo che i cammini minimi siano uguali 
    if logger.isEnabledFor(logging.DEBUG):
        assert dynSssp.distance(to_node) == sssp.distance(to_node)

    if logger.isEnabledFor(logging.DEBUG):
        if(dynSssp.distance(to_node) != sssp.distance(to_node)):
            logger.debug(f"assert error: dynSssp.distance != sssp.distance")

    # controllo che il peso totale dei due grafi sia uguale
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.totalEdgeWeight() == dyn_localGraph.totalEdgeWeight()
    # se i path son diversi e' sufficiente che abbiano la stessa distanza
    #      if logger.isEnabledFor(logging.DEBUG):
    # assert dynSssp.getPath(to_node)  == sssp.getPath(to_node)

# calcola ed inserisce il RunningTime impiegato da Dijkstra e DynDijkstra per calcolare SSSP a fronte di un
# evento EDGE_WEIGHT_INCREMENT - FUNZIONA, NO ERRORI (sssp.distance(to_node) == dynSssp.distance(to_node))
def handleEdgeWeightIncrementEvent(event, localGraph, dyn_localGraph, sssp, dynSssp, static_computing_time_list, dynamic_computing_time_list):
    new_weight = 0
    # cerco un arco random che mi permetta di effettuare l'incremento del peso di almeno 1
    while(True):
        edge_to_change_weight = networkit.graphtools.randomEdge(localGraph, True)
        actual_weight = localGraph.weight(edge_to_change_weight[0], edge_to_change_weight[1])
        if(actual_weight+1 < MAX_EDGE_WEIGHT):
            new_weight = random.randint(actual_weight+1, MAX_EDGE_WEIGHT)
            break
    from_node = edge_to_change_weight[0]
    to_node = edge_to_change_weight[1]
    localGraph.setWeight(from_node, to_node, new_weight)
    dyn_localGraph.setWeight(from_node, to_node, new_weight)
    static_computing_time_list.append(("EDGE_WEIGHT_INCREMENT", computeDijkstra(sssp)))
    # simulo l' incremento del peso usando l'evento EDGE_WEIGHT_UPDATE 
    # poiche' DynDijkstra non supporta l' evento EDGE_WEIGHT_INCREMENT
    dyn_event = networkit.dynamic.GraphEvent(networkit.dynamic.GraphEvent.EDGE_WEIGHT_UPDATE, from_node, to_node, new_weight)
    dynamic_computing_time_list.append(("EDGE_WEIGHT_INCREMENT", computeDynDijkstra(dynSssp, dyn_event)))
    # controllo che l'arco sia stato aggiunto in entrambi i grafi
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.hasEdge(from_node,to_node) == dyn_localGraph.hasEdge(from_node, to_node)
    # controllo che il peso dell'arco aggiunto sia uguale in entrambi i grafi
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.weight(from_node, to_node) == dyn_localGraph.weight(from_node, to_node)
    # controllo che i cammini minimi siano uguali 
    if logger.isEnabledFor(logging.DEBUG):
        assert dynSssp.distance(to_node) == sssp.distance(to_node)
    # controllo che il peso totale dei due grafi sia uguale
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.totalEdgeWeight() == dyn_localGraph.totalEdgeWeight()
    # se i path son diversi e' sufficiente che abbiano la stessa distanza
    #      if logger.isEnabledFor(logging.DEBUG):
    # assert dynSssp.getPath(to_node)  == sssp.getPath(to_node)

# calcola ed inserisce il RunningTime impiegato da Dijkstra e DynDijkstra per calcolare SSSP a fronte di un
# evento EDGE_WEIGHT_DECREMENT
def handleEdgeWeightUpdateEvent(event, localGraph, dyn_localGraph, sssp, dynSssp, static_computing_time_list, dynamic_computing_time_list):
    new_weight = 0
    # cerco un arco random che mi permetta di effettuare il decremento del peso di almeno 1
    while(True):
        edge_to_change_weight = networkit.graphtools.randomEdge(localGraph, True)
        actual_weight = localGraph.weight(edge_to_change_weight[0], edge_to_change_weight[1])
        if(MIN_EDGE_WEIGHT < actual_weight-1):
            new_weight = random.randint(MIN_EDGE_WEIGHT , actual_weight-1)
            break
    from_node = edge_to_change_weight[0]
    to_node = edge_to_change_weight[1]

    localGraph.setWeight(from_node, to_node, new_weight)
    dyn_localGraph.setWeight(from_node, to_node, new_weight)

    static_computing_time_list.append(("EDGE_WEIGHT_DECREMENT", computeDijkstra(sssp)))

    dyn_event = networkit.dynamic.GraphEvent(event, from_node, to_node, new_weight)
    result = computeDynDijkstra(dynSssp, dyn_event)
    dynamic_computing_time_list.append(("EDGE_WEIGHT_DECREMENT", result))

    # controllo che l'arco sia stato aggiunto in entrambi i grafi
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.hasEdge(from_node,to_node) == dyn_localGraph.hasEdge(from_node, to_node)
    # controllo che il peso dell'arco aggiunto sia uguale in entrambi i grafi
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.weight(from_node, to_node) == dyn_localGraph.weight(from_node, to_node)
    # controllo che i cammini minimi siano uguali 
    if logger.isEnabledFor(logging.DEBUG):
        assert dynSssp.distance(to_node) == sssp.distance(to_node)
    # controllo che il peso totale dei due grafi sia uguale
    if logger.isEnabledFor(logging.DEBUG):
        assert localGraph.totalEdgeWeight() == dyn_localGraph.totalEdgeWeight()
    # se i path son diversi e' sufficiente che abbiano la stessa distanza
    #      if logger.isEnabledFor(logging.DEBUG):
    # assert dynSssp.getPath(to_node)  == sssp.getPath(to_node)

def test_DijkstraOnGraphByType(graphType):
    if(isinstance(graphType, GraphTypes) == False):
        return

    parser = graphParser.file_Parser()

    map_result_by_node = {}
    map_result_by_edge = {}

    dyn_map_result_by_node = {}
    dyn_map_result_by_edge = {}

    while(True):      
        response = parser.getNextByType(graphType)

        if(response[0] == "no_more_graphs" or response == "not_exist"):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"break: {response}")
            break
        elif(isinstance(response[0], int) and response[0] <= utility.GRAPH_TO_CHECK - 1):
            index, graph, randomMissingEdgeList = response
            # Static Dijkstra test - worst-case time O(m+n logn) con Fib. Heap
            g0_nodes = graph.numberOfNodes()
            g0_edges = graph.numberOfEdges()

            static_result_list, dynamic_result_list = DijkstraWithRandomEventTest(graph, utility.EVENT_NUMBER_IN_EXP, randomMissingEdgeList)

            # cmp_array = [x[1] for x in static_result_list]

            # np_avg = numpy.average(cmp_array)
            # np_var = numpy.var(cmp_array, dtype=numpy.float64)

            # valutare la media dei rapporti (speedup x ogni esecuzione)
            # e stessa cosa per il cambio del nodo sorgente

            # map_result_by_node[(g0_nodes, g0_edges, g0_totalEdgeWeight)] = (np_avg, np_var)
            # map_result_by_edge[(g0_edges, g0_nodes, g0_totalEdgeWeight)] = (np_avg, np_var)


            # cmp_dyn_array = [x[1] for x in dynamic_result_list]

            # np_dyn_avg = numpy.average(cmp_dyn_array)
            # np_dyn_var = numpy.var(cmp_dyn_array, dtype=numpy.float64)
            
            # dyn_map_result_by_node[(g0_nodes, g0_edges, g0_totalEdgeWeight)] = (np_dyn_avg, np_dyn_var)
            # dyn_map_result_by_edge[(g0_edges, g0_nodes, g0_totalEdgeWeight)] = (np_dyn_avg, np_dyn_var)

            avg_map = {"graph_type" : graphType.Name(),
                        "graph_number" : index,
                        "nodes" : g0_nodes,
                        "edges" : g0_edges,
                        "total_weight" : graph.totalEdgeWeight(),
                        "result_list" : static_result_list}

            dyn_avg_map = {"graph_type" : graphType.Name(),
                        "graph_number" : index,
                        "nodes" : g0_nodes,
                        "edges" : g0_edges,
                        "total_weight" : graph.totalEdgeWeight(),
                        "result_list" : dynamic_result_list}
            
            saveResult(avg_map, utility.ResultType.Static)
            saveResult(dyn_avg_map, utility.ResultType.Dynamic)
        else:
            break

    # plotAll(map_result_by_node, dyn_map_result_by_node, map_result_by_edge,dyn_map_result_by_edge, weighted=True)


def saveResult(map, resultType):
    # tipo del grafo, es: generati da BarabasiAlbert -> BAG, generati da ErdosRenyi -> ERG
    graph_type = map['graph_type']
    # numero del grafo, riferito ai grafi letti da file
    graph_number = map['graph_number']
    # numero di nodi del grafo
    nodes = map['nodes']
    # numero di archi del grafo
    edges = map['edges']
    # (event, running time)
    result_list = map['result_list']


    # columns=['graph_type', 'graph_number', 'nodes', 'edges', 'result_list']
    df = pandas.DataFrame([map])

    if(resultType == utility.ResultType.Static):
        pandas.DataFrame.to_json(df, f"{utility.STATIC_RESULT_FOLDER}/{graph_type}_{graph_number}{utility.RESULT_FILE_TYPE}", indent=4, orient='records')
    elif(resultType == utility.ResultType.Dynamic):
        pandas.DataFrame.to_json(df, f"{utility.DYNAMIC_RESULT_FOLDER}/{graph_type}_{graph_number}{utility.RESULT_FILE_TYPE}", indent=4, orient='records')


if __name__ == "__main__":
    # PROJECT GOAL
    # cambio radice sssp per ammortizzare il bias (10 di cambi)
    # 4 exp per cambio taglia nodi, 4 per archi 
    # 4 cambi di tipologia di grafi (barabasi, erdos, qualche grafo reale)
    # grafici con running time (ordinate) ascisse ( vertici,archi, densita)

    # test_Dijkstra_on_BAGs()
    # test_Dijkstra_on_ERGs()

    utility.clearFolderByType(utility.STATIC_RESULT_FOLDER, GraphTypes.ERG)

    start = time.process_time()
    # test_DijkstraOnGraphByType(GraphTypes.BAG)
    test_DijkstraOnGraphByType(GraphTypes.ERG)
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Experiments ended in {time.process_time() - start} seconds")


    # valutare la media dei rapporti (speedup x ogni esecuzione)
    # e stessa cosa per il cambio del nodo sorgente

    # da valutare perche' gli eventi INCREMENT e EDGE_REMOVAL creano errori su sssp.distance
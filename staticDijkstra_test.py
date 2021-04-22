import logging, time, numpy, random
from mynetworkitfolder import networkit
import matplotlib.pyplot as plt
from networkx.drawing.nx_pylab import draw

logging.basicConfig(filename='staticDijkstra_test.log',  level=logging.DEBUG)

def staticDijkstraTest(g, experiment_number, missing_edge_to_add):
    # Dijkstra’s SSSP algorithm
        # G – networkit.Graph The graph.
        # source – node The source node.
        # storePaths – bool Paths are reconstructable and the number of paths is stored.
        # storeNodesSortedByDistance – bool Store a vector of nodes ordered in increasing distance from the source.
        # target – node target node. Search ends when target node is reached. t is set to None by default.

    for i in range(experiment_number):
        # new local graph instance foreach experiment
        localGraph = g
        
        # first dijkstra computation without adding missing edge
        apsp = networkit.distance.SSSP(g, 0)
        apsp.run()

        # new local missing edge list foreach experiment
        local_missing_edge_array = [edge for edge in missing_edge_to_add]

        # compute dijkstra every time that an edge is added
        for missing_edge in missing_edge_to_add:
            # add one missing edge 
            localGraph.addEdge(0,missing_edge[0],missing_edge[1])

            # compute dijkstra
            local_start_time = time.time()
            apsp = networkit.distance.Dijkstra(g, 0)
            apsp.run()
            local_end_time = time.time()
            # print(f"local computing time: {local_end_time - local_start_time}")
            local_missing_edge_array.remove(missing_edge)

        print(f"Experiment end, edge added, {len(local_missing_edge_array)} remaining")


def dynDijkstraTest(g, experiment_number, missing_edge_to_add):
    # Dijkstra’s SSSP algorithm
        # G – networkit.Graph The graph.
        # source – node The source node.
        # storePaths – bool Paths are reconstructable and the number of paths is stored.
        # storeNodesSortedByDistance – bool Store a vector of nodes ordered in increasing distance from the source.
        # target – node target node. Search ends when target node is reached. t is set to None by default.

    for i in range(experiment_number):
        # new local graph instance foreach experiment
        localGraph = g
        
        # first dijkstra computation without adding missing edge
        sssp = networkit.distance.DynBFS(g, 0)
        sssp.run()

        # new local missing edge list foreach experiment
        local_missing_edge_array = [edge for edge in missing_edge_to_add]

        # compute dijkstra every time that an edge is added
        for missing_edge in missing_edge_to_add:
            # add one missing edge 
            event = localGraph.addEdge(0,missing_edge[0],missing_edge[1])
            print(sssp.getDistances())

            event = networkit.dynamic.GraphEvent(networkit.dynamic.GraphEvent.EDGE_REMOVAL, missing_edge[0],missing_edge[1], 0)
            
            # compute dijkstra
            local_start_time = time.time()
            # sssp.update()
            sssp.update(event)
            local_end_time = time.time()
            # print(f"local computing time: {local_end_time - local_start_time}")
            local_missing_edge_array.remove(missing_edge)

        print(f"Experiment end, edge added, {len(local_missing_edge_array)} remaining")

# non e' fattibile per grafi con piu di 700 nodi
def getMissingEdgeFromGraph(graph):
    missingEdges = []
    for u in graph.iterNodes():
        for v in graph.iterNodes():
            if (u != v and graph.hasEdge(u,v) == False and missingEdges.__contains__((v,u)) == False):
                missingEdges.append((u,v))
    
    # print(missingEdges)
    n = graph.numberOfNodes()
    e = graph.numberOfEdges()
    print(f'#node:{n}, #edge:{e}, #theoreticallyMissingEdges:{(n*(n-1)/2) - e}, #missingEdges:{len(missingEdges)}')
    return missingEdges

def getMissingEdgeRandomlyFromGraph(graph, edge_number):
    missingEdges = []
    nodes = graph.numberOfNodes()

    nodeList = [node for node in range(nodes)]

    # finche' non ho costruito una lista con edge_number archi mancanti
    while(len(missingEdges) < edge_number):
        # prendo due nodi random dall'insieme dei nodi
        edge = random.sample(nodeList, 2)

        # se il grafo non contiene quell'arco E non e' gia' stato aggiunto alla lista degli archi mancanti, lo aggiungo
        if( (graph.hasEdge(edge[0], edge[1]) == False) and missingEdges.__contains__(edge) == False):
            missingEdges.append(edge)

    print(f"archi da trovare: {edge_number}, archi trovati: {len(missingEdges)}")
    n = graph.numberOfNodes()
    e = graph.numberOfEdges()
    print(f'#node:{n}, #edge:{e}, #theoreticallyMissingEdges:{(n*(n-1)/2) - e}, #missingEdges added:{len(missingEdges)}')
    return missingEdges    

if __name__ == "__main__":
    experiment_number = 1

    # BarabasiAlbertGenerator(count k, count nMax, count n0 = 0, bool batagelj = true)
        # k: Number of attachments per node
        # nMax: Number of nodes in the graph
        # n0: Number of connected nodes to begin with
        # batagelj: Specifies whether to use Batagelj and Brandes’s method (much faster) rather than the naive one; default: true
    k = 4
    nMax = 20
    n0 = 0
    batagelj = True
    graph = networkit.generators.BarabasiAlbertGenerator(k, nMax, n0, batagelj).generate()

    edgeToAddDynamically = int(0.8 * nMax * k) # 10% del numero totale di archi 


    # troppo costosa in termini temporali O(n^2)
    # start_computing_missing_edge_time = time.time()
    # missingEdges = getMissingEdgeFromGraph(graph)
    # end_computing_missing_edge_time = time.time()
    # print(f"Computing missing edge ended in {end_computing_missing_edge_time - start_computing_missing_edge_time} seconds")


    # randomicamente invece
    start_computing_missing_edge_time = time.time()
    randomMissingEdgeList = getMissingEdgeRandomlyFromGraph(graph, edgeToAddDynamically)
    end_computing_missing_edge_time = time.time()
    print(f"Random Computing missing edge ended in {end_computing_missing_edge_time - start_computing_missing_edge_time} seconds")

    # sceglie randomicamente un numero edgeToAddDynamically di archi mancanti a g da aggiungere 
    # randomMissingEdgeList = random.sample(missingEdges, edgeToAddDynamically)
    # print(randomMissingEdgeList)

    # Static Dijkstra test 
    start_time = time.time()
    staticDijkstraTest(graph, experiment_number, randomMissingEdgeList)
    end_time = time.time()
    print(f"Static Dijkstra with {experiment_number} experiments ended in {end_time - start_time} seconds")
    print(f"Dijkstra computed foreach dinamically-added edge, total: {len(randomMissingEdgeList)}")


    # Dynamic Dijkstra test 
    start_time = time.time()
    dynDijkstraTest(graph, experiment_number, randomMissingEdgeList)
    end_time = time.time()
    print(f"Dynamic Dijkstra with {experiment_number} experiments ended in {end_time - start_time} seconds")
    print(f"Dijkstra computed foreach dinamically-added edge, total: {len(randomMissingEdgeList)}")

    # OLD CODE - REMOVE?
    # allDijkstraTime = []
    # for i in range(experiment_number):
    #     allDijkstraTime.append(staticDijkstraTest(graph, experiment_number))

    # logging.debug(f'EXPERIMENT Graph: #startingNode:{graph.numberOfNodes()} #startingEdge:{graph.numberOfEdges()} #finalNode:{graph.numberOfNodes()} #finalEdge:{graph.numberOfEdges()+edgeToAddDynamically}')
    # logging.debug(f'DIJKSTRA TIME: avr: {numpy.average(allDijkstraTime)}, var: {numpy.var(allDijkstraTime)}')

    # print(f"startToAPSP: {apsp_time-start_time} , APSPToEnd: {endRun_time-apsp_time} , total: {endRun_time - start_time}")
    # print(f"#nodes: {g.numberOfNodes()}, #edges: {g.numberOfEdges()}")

    # plot graph
    # networkit.viztasks.drawGraph(graph)
    # plt.show()
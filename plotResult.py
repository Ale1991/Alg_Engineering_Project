import pandas, numpy, os
import matplotlib.pyplot as plt
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple
from matplotlib.lines import Line2D
from numpy.core.fromnumeric import sort


from enum import Enum
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

STATIC_RESULT_FOLDER = "Result/Static"
DYNAMIC_RESULT_FOLDER = "Result/Dynamic"
FILE_TYPE = ".json"

# BarabasiAlbertGraph, ErdosRenyiGraph
GRAPH_TYPES = ["BAG", "ERG"]

GRAPH_TO_CHECK = 6
# e'  il numero di eventi randomici che avvengono ad ogni esperimento di dijkstra (per ogni grafo)
EVENT_NUMBER_IN_EXP = 1999

def saveStaticResult(map):
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
    pandas.DataFrame.to_json(df, f"Result/Static/{graph_type}_{graph_number}.json", indent=4, orient='records')

    # dic = pandas.read_json('Result/ERG_0.json')
    # print("read")

def saveDynamicResult(map):
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
    pandas.DataFrame.to_json(df, f"Result/Dynamic/{graph_type}_{graph_number}.json", indent=4, orient='records')

    # dic = pandas.read_json('Result/ERG_0.json')
    # print("read")

# algo_type = static/dynamic
# type = GRAPH_TYPES
# index = graph number
def readResultFromFile(algo_type, graph_type, index):
    folder = ""
    if(algo_type == DijkstraAlgoTypes.STATIC):
        folder = STATIC_RESULT_FOLDER
    elif(algo_type == DijkstraAlgoTypes.DYNAMIC):
        folder = DYNAMIC_RESULT_FOLDER
    else:
        pass # error

    searchingFile = (graph_type.Name() + "_" + index.__str__() + FILE_TYPE)

    for root, dirs, files in os.walk(folder):
        for file in files:
            if(file == searchingFile):
                path = os.path.join(root, file)
                if(os.path.isfile(path)):
                    result = pandas.read_json(path)
                    return result
                else:
                    return "not_exist"


def plotResultList(index, static_result_list, dynamic_result_list):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f'Graph_{index} => array_len vs running_time', fontsize=20)
    x_st_arr = [x for x in range(len(static_result_list))]
    x_dyn_arr = [x for x in range(len(dynamic_result_list))]
    y_st_arr = [x[1] for x in static_result_list]
    y_dyn_arr = [x[1] for x in dynamic_result_list]

    assert x_st_arr == x_dyn_arr

    st_var = numpy.var(y_st_arr)
    dyn_var = numpy.var(y_dyn_arr)

    st_std = numpy.std(y_st_arr)
    dyn_std = numpy.std(y_dyn_arr)

    st_avg = numpy.average(y_st_arr)
    dyn_avg = numpy.average(y_dyn_arr)

    ax.plot(x_st_arr, y_st_arr, marker="o", label= "staticDijkstraRT")
    ax.plot(x_dyn_arr, y_dyn_arr, marker="o", label= "dynDijkstraRT")
    ax.legend()

def plotNodiRT(static_result_map, dyn_result_map):
    # Map risultati Dijkstra: key=(nodi, archi) : val=(media_RT, varianza_RT)
    static_sorted_map_by_nodes = {}
    for key in sorted(static_result_map):
        static_sorted_map_by_nodes[key] = static_result_map[key]

    # Map risultati DynDijkstra: key=(nodi, archi) : val=(media_RT, varianza_RT)
    dyn_sorted_map_by_nodes = {}
    for key in sorted(dyn_result_map):
        dyn_sorted_map_by_nodes[key] = dyn_result_map[key]

    # costruisco un array ordinato al crescere dei nodi con valori (nodi, avg_computing_time)
    static_node_avg_arr = []
    for k in static_sorted_map_by_nodes:

        static_node_avg_arr.append((k[0],static_sorted_map_by_nodes[k][0]))

    # costruisco un array ordinato al crescere dei nodi con valori (nodi, avg_computing_time)
    dyn_node_avg_arr = []
    for k in dyn_sorted_map_by_nodes:
        dyn_node_avg_arr.append((k[0],dyn_sorted_map_by_nodes[k][0]))

    static_nodes = [x[0] for x in static_node_avg_arr]
    static_RT = [x[1] for x in static_node_avg_arr]
    dyn_nodes = [x[0] for x in dyn_node_avg_arr]
    dyn_RT = [x[1] for x in dyn_node_avg_arr]

    assert static_nodes == dyn_nodes

    fig, ax = plt.subplots(2,1, figsize=(10, 10))
    fig.suptitle('Nodes vs RunningTime', fontsize=20)
    # SUBPLOT 1
    ax[0].set_xlabel('nodes', fontsize=10)
    ax[0].set_ylabel('running_time', fontsize=10)

    ax[0].plot(static_nodes, static_RT, marker="o", label= "staticDijkstra")
    ax[0].plot(dyn_nodes, dyn_RT, marker="o", label= "dynDijkstra")
    ax[0].legend()

    # SUBPLOT 2
    ax[1].set_xlabel('nodes', fontsize=10)
    ax[1].set_ylabel('running_time', fontsize=10)
    ax[1].plot(dyn_nodes, dyn_RT, marker="o", label= "dynDijkstra")
    ax[1].legend()

def plotArchiRT(static_result_map, dyn_result_map):
    # sort static result by edges
    static_sorted_map_by_edges = {}
    for key in sorted(static_result_map):
        static_sorted_map_by_edges[key] = static_result_map[key]

    # sort dynamic result by edges
    dyn_sorted_map_by_edges = {}
    for key in sorted(dyn_result_map):
        dyn_sorted_map_by_edges[key] = dyn_result_map[key]

    # costruisco un array ordinato al crescere degli archi con valori (archi, avg_computing_time)
    edge_avg_arr = []
    for k in static_sorted_map_by_edges:
        edge_avg_arr.append((k[0], static_sorted_map_by_edges[k][0]))

    # costruisco un array ordinato al crescere degli archi con valori (archi, avg_computing_time)
    dyn_edge_avg_arr = []
    for k in dyn_sorted_map_by_edges:
        dyn_edge_avg_arr.append((k[0], dyn_sorted_map_by_edges[k][0]))

    static_edges = [x[0] for x in edge_avg_arr]
    static_RT = [x[1] for x in edge_avg_arr]
    dyn_edges = [x[0] for x in dyn_edge_avg_arr]
    dyn_RT = [x[1] for x in dyn_edge_avg_arr]

    assert static_edges == dyn_edges

    fig, ax = plt.subplots(2,1, figsize=(10, 10))
    fig.suptitle('Edges vs RunningTime', fontsize=20)
    # SUBPLOT 1
    ax[0].set_xlabel('edges', fontsize=10)
    ax[0].set_ylabel('running_time', fontsize=10)

    ax[0].plot(static_edges, static_RT, marker="o", label= "staticDijkstra")
    ax[0].plot(dyn_edges, dyn_RT, marker="o", label= "dynDijkstra")
    ax[0].legend()

    # SUBPLOT 2
    ax[1].set_xlabel('edges', fontsize=10)
    ax[1].set_ylabel('running_time', fontsize=10)
    ax[1].plot(dyn_edges, dyn_RT, marker="o", label= "dynDijkstra")
    ax[1].legend()

# NON USATO - Plot #Archi/#Nodi sulle ascisse e RunningTime sulle ordinate
def plotEdgesDivNodesRT(static_result_map, dyn_result_map):
    # sort static result by edges
    static_sorted_map_by_edges = {}
    for key in sorted(static_result_map):
        static_sorted_map_by_edges[key] = static_result_map[key]

    # sort dynamic result by edges
    dyn_sorted_map_by_edges = {}
    for key in sorted(dyn_result_map):
        dyn_sorted_map_by_edges[key] = dyn_result_map[key]

    # costruisco un array ordinato al crescere del rapporto #archi/#nodi con valori (#archi/#nodi, avg_computing_time)
    edge_node_ratio_avg_arr = []
    for k in static_sorted_map_by_edges:
        edge_node_ratio_avg_arr.append((k[0]/k[1], static_sorted_map_by_edges[k][0]))

    # costruisco un array ordinato al crescere del rapporto #archi/#nodi con valori (#archi/#nodi, avg_computing_time)
    dyn_edge_node_ratio_avg_arr = []
    for k in dyn_sorted_map_by_edges:
        dyn_edge_node_ratio_avg_arr.append((k[0]/k[1], dyn_sorted_map_by_edges[k][0]))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('#Edges/#Nodes vs RunningTime', fontsize=20)
    # ordino per archi crescenti
    edge_node_ratio_avg_arr.sort()
    dyn_edge_node_ratio_avg_arr.sort()
    plt.subplot(221)
    plt.xlabel('#edge/#node', fontsize=8)
    plt.ylabel('running_time', fontsize=8)
    plt.plot([x[0] for x in edge_node_ratio_avg_arr], [x[1] for x in edge_node_ratio_avg_arr], marker="o")
    plt.plot([x[0] for x in dyn_edge_node_ratio_avg_arr], [x[1] for x in dyn_edge_node_ratio_avg_arr], marker="o")
    plt.subplot(222)
    plt.plot([x[0] for x in dyn_edge_node_ratio_avg_arr], [x[1] for x in dyn_edge_node_ratio_avg_arr], marker="o")

# FATTO
def plotAsympNot(static_result_map, dyn_result_map):

    # Map risultati Dijkstra: key=(nodi, archi) : val=(media_RT, varianza_RT)
    static_sorted_map_by_nodes = {}
    for key in sorted(static_result_map):
        static_sorted_map_by_nodes[key] = static_result_map[key]

    # Map risultati DynDijkstra: key=(nodi, archi) : val=(media_RT, varianza_RT)
    dyn_sorted_map_by_nodes = {}
    for key in sorted(dyn_result_map):
        dyn_sorted_map_by_nodes[key] = dyn_result_map[key]

    import math
    val_arr = []
    for x in static_sorted_map_by_nodes:
        n = x[0]
        e = x[1]
        avg = static_sorted_map_by_nodes[x][0]
        res = e + n * math.log(n)
        val_arr.append((res, avg))
    val_arr.sort()

    dyn_val_arr = []
    for x in dyn_sorted_map_by_nodes:
        n = x[0]
        e = x[1]
        avg = dyn_sorted_map_by_nodes[x][0]
        res = e + n * math.log(n)
        dyn_val_arr.append((res, avg))
    dyn_val_arr.sort()

    static_asymp = [x[0] for x in val_arr]
    static_RT = [x[1] for x in val_arr]
    dyn_asymp = [x[0] for x in dyn_val_arr]
    dyn_RT = [x[1] for x in dyn_val_arr]
    assert static_asymp == dyn_asymp

    fig, ax = plt.subplots(2,1, figsize=(10, 10))
    fig.suptitle('Asymptotic analysis vs RunningTime', fontsize=20)
    
    # SUBPLOT 1
    ax[0].set_xlabel('m + n * log(n)', fontsize=10)
    ax[0].set_ylabel('running_time', fontsize=10)

    ax[0].plot(static_asymp, static_RT, marker="o", label= "staticDijkstra")
    ax[0].plot(dyn_asymp, dyn_RT, marker="o", label= "dynDijkstra")
    ax[0].legend()

    # SUBPLOT 2
    ax[1].set_xlabel('m + n * log(n)', fontsize=10)
    ax[1].set_ylabel('running_time', fontsize=10)
    ax[1].plot(dyn_asymp, dyn_RT, marker="o", label= "dynDijkstra")
    ax[1].legend()

#Densita' grafo non orientato: 2m/n*(n-1)
#Densita' grafo PESATO non orientato: 2w/n*(n-1)
# FATTO
def plotDensityRT(static_result_map, dyn_result_map, weighted):
    # costruisco un array con valori (densita', avg_running_time)
    static_density_avg_arr = []
    for k in static_result_map:
        n = k[0]
        m = k[1]
        w = k[2]
        if(weighted):
            density = (2*w)/(n*(n-1))
        else:
            density = (2*m)/(n*(n-1))
        static_density_avg_arr.append((density, static_result_map[k][0]))

    # costruisco un array con valori (densita', avg_running_time)
    dyn_density_avg_arr = []
    for k in dyn_result_map:
        n = k[0]
        m = k[1]
        w = k[2]
        if(weighted):
            density = (2*w)/(n*(n-1))
        else:
            density = (2*m)/(n*(n-1))
        dyn_density_avg_arr.append((density, dyn_result_map[k][0]))

    fig, ax = plt.subplots(2,1, figsize=(10, 10))
    if(weighted):
        fig.suptitle('Weighted_Density vs RunningTime', fontsize=20)
    else:
        fig.suptitle('Density vs RunningTime', fontsize=20)
    
    # ordino per densita' crescente
    static_density_avg_arr.sort()
    dyn_density_avg_arr.sort()
    
    density_arr = [x[0] for x in static_density_avg_arr]
    dyn_density_arr = [x[0] for x in dyn_density_avg_arr]
    static_RT_arr = [x[1] for x in static_density_avg_arr]
    dyn_RT_arr = [x[1] for x in dyn_density_avg_arr]

    # controllo se la densita' e' uguale per entrambi i plot negli stessi punti
    assert density_arr == dyn_density_arr

    # SUBPLOT 1
    if(weighted):
        ax[0].set_xlabel('2(sum(w))/n*(n-1)', fontsize=10)
    else:
        ax[0].set_xlabel('2*m)/n*(n-1)', fontsize=10)
    ax[0].set_ylabel('running_time', fontsize=10)

    ax[0].plot(density_arr, static_RT_arr, marker="o", label= "staticDijkstra")
    ax[0].plot(density_arr, dyn_RT_arr, marker="o", label= "dynDijkstra")
    ax[0].legend()

    # SUBPLOT 2
    if(weighted):
        ax[1].set_xlabel('2(sum(w))/n*(n-1)', fontsize=10)
    else:
        ax[1].set_xlabel('2*m)/n*(n-1)', fontsize=10)
    ax[1].set_ylabel('running_time', fontsize=10)
    ax[1].plot(dyn_density_arr, dyn_RT_arr, marker="o", label= "dynDijkstra")
    ax[1].legend()

# FATTO
def plotDensitySpeedUp(static_result_map, dyn_result_map, weighted):
    # costruisco un array con valori (densita', avg_running_time)
    static_density_avg_arr = []
    for k in static_result_map:
        n = k[0]
        m = k[1]
        w = k[2]
        if(weighted):
            density = ((2*w)/(n*(n-1)))
        else:
            density = (2*m)/(n*(n-1))
        static_density_avg_arr.append((density, static_result_map[k][0]))

    # costruisco un array con valori (densita', avg_running_time)
    dyn_density_avg_arr = []
    for k in dyn_result_map:
        n = k[0]
        m = k[1]
        w = k[2]
        if(weighted):
            density = ((2*w)/(n*(n-1)))
        else:
            density = (2*m)/(n*(n-1))
        dyn_density_avg_arr.append((density, dyn_result_map[k][0]))

    fig, ax = plt.subplots(figsize=(10, 10))

    if(weighted):
        fig.suptitle('Weighted_Density vs SpeedUp', fontsize=20)
    else:
        fig.suptitle('Density vs SpeedUp', fontsize=20)
    
    # ordino per densita' crescente
    static_density_avg_arr.sort()
    dyn_density_avg_arr.sort()

    speed_up = []

    for k in range(len(static_density_avg_arr)):
        st = static_density_avg_arr[k]
        dy = dyn_density_avg_arr[k]

        if(k > 0):
            # mi assicuro che l'array sia ordinato per densita' crescente
            assert static_density_avg_arr[k][0] > static_density_avg_arr[k-1][0]
            assert dyn_density_avg_arr[k][0] > dyn_density_avg_arr[k-1][0]

        # controllo se sto prendendo correttamente i runningtime riferiti alla stessa densita' 
        assert st[0] == dy[0]

        speed_up.append(st[1]/dy[1])
    
    density_arr = [x[0] for x in static_density_avg_arr]
    dyn_density_arr = [x[0] for x in dyn_density_avg_arr]

    # controllo se la densita' e' uguale per entrambi i plot negli stessi punti
    assert density_arr == dyn_density_arr

    # SUBPLOT 1
    if(weighted):
        ax.set_xlabel('2(sum(w))/n*(n-1)', fontsize=10)
    else:
        ax.set_xlabel('2*m)/n*(n-1)', fontsize=10)
    ax.set_ylabel('SpeedUp', fontsize=10)

    ax.plot(density_arr, speed_up, marker="o", label= "staticDijkstraRT/dynDijkstraRT")
    ax.legend()

# FATTO
def plotNodeSpeedUp(static_result_map, dyn_result_map):
    st_x = [x[0] for x in static_result_map]
    dyn_x = [x[0] for x in dyn_result_map]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Nodes Vs SpeedUp', fontsize=20)

    speed_up = []

    for k in static_result_map:
        st = static_result_map[k]
        dy = dyn_result_map[k]

        speed_up.append(st[0]/dy[0])

    # controllo se la densita' e' uguale per entrambi i plot negli stessi punti
    assert st_x == dyn_x

    # SUBPLOT 1
    ax.set_xlabel('Nodes', fontsize=10)
    ax.set_ylabel('SpeedUp', fontsize=10)

    ax.plot(st_x, speed_up, marker="o", label= "staticDijkstraRT/dynDijkstraRT")
    ax.legend()

def plotAllSpeedUp(speedUp_list):

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('event vs speedup', fontsize=20)

    # SUBPLOT 1
    ax.set_xlabel('event', fontsize=10)
    ax.set_ylabel('speedup', fontsize=10)
    x = [x for x in range(len(speedUp_list))]
    plt.scatter(x, speedUp_list)
    model = numpy.polyfit(x, speedUp_list, 10)

    predict = numpy.poly1d(model)
    x_lin_reg = range(len(speedUp_list))
    y_lin_reg = predict(x_lin_reg)
    plt.scatter(x, speedUp_list)
    plt.plot(x_lin_reg, y_lin_reg, c = 'r')

    # ax.plot(x, speedUp_list, marker="o", label= "staticDijkstraRT/dynDijkstraRT")
    # ax.legend()
    plt.show()
# # # # # # # # #
# PLOT FUNCTION #
# # # # # # # # #
def plotAll(map_result_by_node, dyn_map_result_by_node, map_result_by_edge,dyn_map_result_by_edge, weighted):

    # plotNodiRT(map_result_by_node, dyn_map_result_by_node)

    # plotArchiRT(map_result_by_edge,dyn_map_result_by_edge)

    # plotAsympNot(map_result_by_node, dyn_map_result_by_node)

    # plotDensityRT(map_result_by_node,dyn_map_result_by_node, True) # weighted = True per grafi pesati

    plotDensitySpeedUp(map_result_by_node, dyn_map_result_by_node, weighted)

    plotNodeSpeedUp(map_result_by_node, dyn_map_result_by_node)

    plt.show()
    print("Test end")

if __name__ == "__main__":
    # PROJECT GOAL
    # cambio radice sssp per ammortizzare il bias (10 di cambi)
    # 4 exp per cambio taglia nodi, 4 per archi 
    # 4 cambi di tipologia di grafi (barabasi, erdos, qualche grafo reale)
    # grafici con running time (ordinate) ascisse ( vertici,archi, densita)
    # plotAll()
    map_result_by_node = {}
    map_result_by_edge = {}

    dyn_map_result_by_node = {}
    dyn_map_result_by_edge = {}

    avg_SpeedUp = []
    len_counter = 0
    for index in range(GRAPH_TO_CHECK):

        # da qui ciclo per tutti i file per ricostruire le dyn_map_result_by_nodes da dare in pasto alle plot func
        result_map = readResultFromFile(DijkstraAlgoTypes.STATIC, GraphTypes.ERG, index)
        graph_type = result_map['graph_type'].item()
        graph_number = result_map['graph_number'].item()
        nodes = result_map['nodes'].item()
        edges = result_map['edges'].item()
        total_weight = result_map['total_weight'].item()
        static_result_list = result_map['result_list'].item()

        cmp_array = [x[1] for x in static_result_list]
        np_avg = numpy.average(cmp_array)
        np_var = numpy.var(cmp_array, dtype=numpy.float64)

        # valutare la media dei rapporti (speedup x ogni esecuzione)
        # e stessa cosa per il cambio del nodo sorgente

        map_result_by_node[(nodes, edges, total_weight)] = (np_avg, np_var)
        map_result_by_edge[(edges, nodes, total_weight)] = (np_avg, np_var)


        dyn_result_map = readResultFromFile(DijkstraAlgoTypes.DYNAMIC, GraphTypes.ERG, index)
        graph_type = dyn_result_map['graph_type'].item()
        graph_number = dyn_result_map['graph_number'].item()
        nodes = dyn_result_map['nodes'].item()
        edges = dyn_result_map['edges'].item()
        total_weight = dyn_result_map['total_weight'].item()
        dynamic_result_list = dyn_result_map['result_list'].item()

        cmp_dyn_array = [x[1] for x in dynamic_result_list]
        np_dyn_avg = numpy.average(cmp_dyn_array)
        np_dyn_var = numpy.var(cmp_dyn_array, dtype=numpy.float64)

        dyn_map_result_by_node[(nodes, edges, total_weight)] = (np_dyn_avg, np_dyn_var)
        dyn_map_result_by_edge[(edges, nodes, total_weight)] = (np_dyn_avg, np_dyn_var)

        for i in range(len(dynamic_result_list)):
            if(dynamic_result_list[i][0] == static_result_list[i][0]):
                speedup = static_result_list[i][1]/dynamic_result_list[i][1]
                avg_SpeedUp.append(speedup)

        len_counter += len(dynamic_result_list)
        if(len(avg_SpeedUp) == len_counter):
            print(f"ok for G{index}")
        else:
            print(f"something went wrong for G{index}")

        

    plotAllSpeedUp(avg_SpeedUp)
    print('finish')
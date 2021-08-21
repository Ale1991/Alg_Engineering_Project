import pandas, numpy, os, sys, logging, utility
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("plotResult")
logger.setLevel(utility.DEBUG)

# algo_type = static/dynamic
# type = GraphTypes
# index = graph number
def readResultFromFile(algo_type, graph_type, index):
    folder = ""
    if(algo_type == utility.DijkstraAlgoTypes.STATIC):
        folder = utility.STATIC_RESULT_FOLDER
    elif(algo_type == utility.DijkstraAlgoTypes.DYNAMIC):
        folder = utility.DYNAMIC_RESULT_FOLDER
    else:
        pass # error

    searchingFile = (graph_type.Name() + "_" + index.__str__() + utility.RESULT_FILE_TYPE)

    for root, dirs, files in os.walk(folder):
        for file in files:
            if(file == searchingFile):
                path = os.path.join(root, file)
                if(os.path.isfile(path)):
                    result = pandas.read_json(path)
                    return result
                else:
                    return "not_exist"

def GetGraphDataStructByType(graphType):
    if(isinstance(graphType, utility.GraphTypes) == False):
        return

    data_struct = []
    for index in range(utility.GRAPH_TO_CHECK):
        # da qui ciclo tutti i file per ricostruire una lista
        # data_struct = [[graph_type, graph_number, nodes, edges, total_weight, static_rt, dyn_rt]]
        # da dare in pasto alle plot func
        static_result_map = readResultFromFile(utility.DijkstraAlgoTypes.STATIC, graphType, index)
        dyn_result_map = readResultFromFile(utility.DijkstraAlgoTypes.DYNAMIC, graphType, index)

        if(isinstance(static_result_map, DataFrame) == False):
            logger.debug(f"File {graphType}_{index} for {utility.DijkstraAlgoTypes.STATIC} not found")
        elif(isinstance(dyn_result_map, DataFrame) == False):
            logger.debug(f"File {graphType}_{index} for {utility.DijkstraAlgoTypes.DYNAMIC} not found")
        else:     
            graph_type = static_result_map['graph_type'].item()
            graph_number = static_result_map['graph_number'].item()
            nodes = static_result_map['nodes'].item()
            edges = static_result_map['edges'].item()
            total_weight = static_result_map['total_weight'].item()

            static_result_list = static_result_map['result_list'].item()
            dynamic_result_list = dyn_result_map['result_list'].item()

            if logger.isEnabledFor(logging.DEBUG):
                assert graph_type == dyn_result_map['graph_type'].item()
                assert graph_number == dyn_result_map['graph_number'].item()
                assert nodes == dyn_result_map['nodes'].item()
                assert edges == dyn_result_map['edges'].item()
                assert total_weight == dyn_result_map['total_weight'].item()
                print(f"all assert passed for {graph_type}_{graph_number}")

            static_rt = [x[1] for x in static_result_list]
            dyn_rt = [x[1] for x in dynamic_result_list]
            data_struct.append([graph_type, graph_number, nodes, edges, total_weight, static_rt, dyn_rt])        

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Parsing Files End')

    return data_struct


# # # # # # # # #
# PLOT FUNCTION #
# # # # # # # # #
# data_struct = [[graph_type, graph_number, nodes, edges, total_weight, static_rt, dyn_rt]]
def plotAll(data_struct):

    plotAsymptoticRT(data_struct)
    plotDensityRT(data_struct)
    plotNodesSpeedUp(data_struct)
    plotAllSpeedUp(data_struct)

    plt.show()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Test end")

# ascisse => Numero degli eventi di modifica dei grafi
# ordinate => Tutti speedup(running_time_Dijkstra/running_time_DynDijkstra) calcolati per ogni evento
# durante un'ispezione visiva dei punti e' emersa una relazione curvilinea tra le variabili => regressione polinomiale
def plotAllSpeedUp(data_struct):
    # lista contenente una lista di speedup per ogni grafo
    speedUp_list = []

    for tup in data_struct:
        st_list = tup[5]
        dyn_list = tup[6]
        assert len(st_list) == len(dyn_list)
        # calcolo lo speedup per ogni evento
        for i in range(len(st_list)):
            speedUp_list.append(st_list[i]/dyn_list[i])

        #speedUp_list.append([st_list[x]/dyn_list[x] for x in range(len(st_list))])

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
    plt.scatter(x, speedUp_list, label="speedup after a single event")
    line, = plt.plot(x_lin_reg, y_lin_reg, label="polynomial regression model", c = 'b')
    legend = ax.legend(loc='upper left')
    # points_legend = ax.legend(handles=[points], loc='center right')

# ascisse => Numero dei nodi del grafo
# ordinate => media degli speedup(running_time_Dijkstra/running_time_DynDijkstra) per ogni grafo
def plotNodesSpeedUp(data_struct):
    # lista contenente una lista di speedup per ogni grafo
    speedup_list = []

    for tup in data_struct:
        st_list = tup[5]
        dyn_list = tup[6]
        assert len(st_list) == len(dyn_list)
        # calcolo lo speedup per ogni evento
        speedup_list.append([st_list[x]/dyn_list[x] for x in range(len(st_list))])

    # lista contenente la media degli speedup per ogni grafo
    all_graph_speedup = [numpy.average(x) for x in speedup_list]

    # lista contenente il numero dei nodi di ogni grafo
    x = [tuple[2] for tuple in data_struct]

    fig, ax = plt.subplots()
    fig.suptitle('#Node vs Speedup', fontsize=20)
    ax.set_xlabel('#Nodes', fontsize=10)
    ax.set_ylabel('Speedup', fontsize=10)  

    line, = ax.plot(x, all_graph_speedup, label="line", marker="o", c = 'r')
    # legend = ax.legend(handles=[line], loc='center right')

# ascisse => Densita' grafo non orientato: 2m/n*(n-1) # Densita' grafo PESATO non orientato: 2w/n*(n-1)
# ordinate => media degli speedup(running_time_Dijkstra/running_time_DynDijkstra) per ogni grafo
def plotDensitySpeedUp(data_struct):
    weighted = True

    # costruisco una lista con [index, densita']
    index_density_speedup_list = []
    for tup in data_struct:
        index = tup[1]
        n = tup[2]
        m = tup[3]
        w = tup[4]
        if(weighted):
            density = ((2*w)/(n*(n-1)))
        else:
            density = (2*m)/(n*(n-1))
        
        local_speedup = []
        static_list = tup[5]
        dynamic_list = tup[6]

        assert len(static_list) == len(dynamic_list)

        for i in range(len(static_list)):
            local_speedup.append(static_list[i]/dynamic_list[i])

        index_density_speedup_list.append([index, density, local_speedup])

    
    # density list
    x = [x[1] for x in index_density_speedup_list]
    # average speedup list
    y = [numpy.average(x[2]) for x in index_density_speedup_list]
    test = []

    for i in range(len(x)):
        test.append([x[i], y[i]])
    test.sort()

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('density vs speedup', fontsize=20)
    # SUBPLOT 1
    if(weighted):
        ax.set_xlabel('2(sum(w))/n*(n-1)', fontsize=10)
    else:
        ax.set_xlabel('2*m)/n*(n-1)', fontsize=10)
    ax.set_ylabel('SpeedUp', fontsize=10)
    x = [x[0] for x in test]
    y = [x[1] for x in test]
    plt.plot(x, y, marker="o", c = 'r')

# ascisse => Densita' grafo non orientato: 2m/n*(n-1) # Densita' grafo PESATO non orientato: 2w/n*(n-1)
# ordinate => media dei Running Time dei due algoritmi per ogni grafo
def plotDensityRT(data_struct):
    weighted = True
    # costruisco una lista con [index, st_RT, dyn_RT]
    density_index_stRT_dynRT_list = []
    for tup in data_struct:
        index = tup[1]
        n = tup[2]
        m = tup[3]
        w = tup[4]
        if(weighted):
            density = ((2*w)/(n*(n-1)))
        else:
            density = (2*m)/(n*(n-1))

        # controllo che la lunghezza della lista contenente i running time relativi all'algo di Dijkstra sia uguale
        # a quella relativa a DynDijkstra
        assert len(tup[5]) == len(tup[6])

        density_index_stRT_dynRT_list.append([density, index, numpy.average(tup[5]), numpy.average(tup[6])])

    density_index_stRT_dynRT_list.sort()

    # density list
    x = [x[0] for x in density_index_stRT_dynRT_list]
    # average running time list
    st_y = [x[2] for x in density_index_stRT_dynRT_list]
    dyn_y = [x[3] for x in density_index_stRT_dynRT_list]

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('density vs running time', fontsize=20)
    # SUBPLOT 1
    if(weighted):
        ax1.set_xlabel('2(sum(w))/n*(n-1)', fontsize=10)
    else:
        ax1.set_xlabel('2*m)/n*(n-1)', fontsize=10)
    ax1.set_ylabel('running time', fontsize=10)

    line1, = ax1.plot(x, st_y, label="Dijkstra", marker="o", c = 'b')
    line2, = ax1.plot(x, dyn_y, label="DynDijkstra", marker="*", c = 'r')
    first_legend = ax1.legend(handles=[line1, line2], loc='center right')

    # fig2, ax2 = plt.subplots(figsize=(10, 10))
    # fig2.suptitle('density vs running time', fontsize=20)
    # SUBPLOT 2
    if(weighted):
        ax2.set_xlabel('2(sum(w))/n*(n-1)', fontsize=10)
    else:
        ax2.set_xlabel('2*m)/n*(n-1)', fontsize=10)
    ax2.set_ylabel('running time', fontsize=10)

    line3, = ax2.plot(x, dyn_y, label="DynDijkstra", marker="*", c = 'r')
    second_legend = ax2.legend(handles=[line3], loc='center right')

# ascisse => m + n log n
# ordinate => media dei Running Time dei due algoritmi per ogni grafo
def plotAsymptoticRT(data_struct):
    import math

    index_asym_stRT_dynRT_list = []
    for tup in data_struct:
        index = tup[1]
        n = tup[2]
        m = tup[3]
        w = tup[4]

        asym =  m + n * math.log(n)

        # controllo che la lunghezza della lista contenente i running time relativi all'algo di Dijkstra sia uguale
        # a quella relativa a DynDijkstra
        assert len(tup[5]) == len(tup[6])

        factor = 1 
        index_asym_stRT_dynRT_list.append([index, asym, numpy.average(tup[5])/factor, numpy.average(tup[6])/factor])

    # index_asym_stRT_dynRT_list.sort()

    # density list
    x = [x[1] for x in index_asym_stRT_dynRT_list]
    # average running time list
    st_y = [x[2] for x in index_asym_stRT_dynRT_list]
    dyn_y = [x[3] for x in index_asym_stRT_dynRT_list]

    # SUBPLOT 1
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Expected Asymptotic Notation vs Running Time', fontsize=20)  
    ax1.set_xlabel('m + n log n', fontsize=10)
    ax1.set_ylabel('running time', fontsize=10)
    line1, = ax1.plot(x, st_y, label="Dijkstra", marker="o", c = 'b')
    line2, = ax1.plot(x, dyn_y, label="DynDijkstra", marker="*", c = 'r')

    first_legend = ax1.legend(handles=[line1, line2], loc='center right')
    #second_legend = plt.legend(handles=[], loc='upper right')
    # SUBPLOT 2
    # fig, ax = plt.subplots(figsize=(10, 10))
    # fig.suptitle('Expected Asymptotic Notation vs Running Time', fontsize=20) 
    ax2.set_xlabel('m + n log n', fontsize=10)
    ax2.set_ylabel('running time', fontsize=10)
    line3, = ax2.plot(x, dyn_y, label="DynDijkstra", marker="*", c = 'r')
    second_legend = ax2.legend(handles=[line3], loc='center right')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

if __name__ == "__main__":
    # PROJECT GOAL
    # cambio radice sssp per ammortizzare il bias (10 di cambi)
    # 4 exp per cambio taglia nodi, 4 per archi 
    # 4 cambi di tipologia di grafi (barabasi, erdos, qualche grafo reale)
    # grafici con running time (ordinate) ascisse ( vertici,archi, densita)
    data_struct = GetGraphDataStructByType(utility.GraphTypes.BAG)
    plotAll(data_struct)

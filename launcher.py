import argparse, utility, sys

parser = argparse.ArgumentParser(prog='launcher.py', description="Dijkstra versus DynDijkstra")
parser.add_argument( '-g', '--genGraphs',   nargs='?', metavar='GraphType', choices=('BAG','ERG'), help='generate files containing graph with types: {%(choices)s}')
parser.add_argument( '-r', '--runDijkstra', nargs='?', metavar='GraphType', choices=('BAG','ERG'), help='run Dijkstra & DynDijkstra Alg. on graph with types: {%(choices)s}')
parser.add_argument( '-p', '--plotResults', nargs='?', metavar='GraphType', choices=('BAG','ERG'), help='plot results obtained from Dijkstra & DynDijkstra Alg. on graph with types: {%(choices)s}')

def main():
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return # sys.exit()

    genGraphs = args.genGraphs
    runDijkstra = args.runDijkstra
    plotResults = args.plotResults

    import time
    start = time.process_time()
    if(genGraphs != None):
        import genGraph
        if genGraphs == 'BAG':
            utility.clearFolderByType(utility.BAGs_FOLDER , utility.GraphTypes.BAG)
            genGraph.genGraphByType(utility.GraphTypes.BAG)
        elif genGraphs == 'ERG':
            utility.clearFolderByType(utility.ERGs_FOLDER, utility.GraphTypes.ERG)
            genGraph.genGraphByType(utility.GraphTypes.ERG)
        print(f"{genGraphs} Graphs generated")
    if(runDijkstra != None):
        import testDijkstra
        if runDijkstra == 'BAG':
            utility.clearFolderByType(utility.STATIC_RESULT_FOLDER, utility.GraphTypes.BAG)
            utility.clearFolderByType(utility.DYNAMIC_RESULT_FOLDER, utility.GraphTypes.BAG)
            testDijkstra.test_DijkstraOnGraphByType(utility.GraphTypes.BAG)
        elif runDijkstra == 'ERG':
            utility.clearFolderByType(utility.STATIC_RESULT_FOLDER, utility.GraphTypes.ERG)
            utility.clearFolderByType(utility.DYNAMIC_RESULT_FOLDER, utility.GraphTypes.ERG)
            testDijkstra.test_DijkstraOnGraphByType(utility.GraphTypes.ERG)
        print(f"{genGraphs} Dijkstra computed")
    if(plotResults != None):
        import plotResult
        data_struct = []
        if plotResults == 'BAG':
            data_struct = plotResult.GetGraphDataStructByType(utility.GraphTypes.BAG)
        elif plotResults == 'ERG':
            data_struct = plotResult.GetGraphDataStructByType(utility.GraphTypes.ERG)
        plotResult.plotAll(data_struct)
        print(f"{genGraphs} Plot ended")
    print(f"test ended in {time.process_time() - start} seconds")
    


if __name__ == '__main__':
    main()
    
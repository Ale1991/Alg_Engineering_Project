import argparse, utility

parser = argparse.ArgumentParser(prog='launcher.py', description="Dijkstra versus DynDijkstra")
parser.add_argument( '-g', '--genGraphs',   nargs='?', metavar='GraphType', choices=('BAG','ERG'), help='generate files containing graph with types: {%(choices)s}')
parser.add_argument( '-r', '--runDijkstra', nargs='?', metavar='GraphType', choices=('BAG','ERG'), help='run Dijkstra & DynDijkstra Alg. on graph with types: {%(choices)s}')
parser.add_argument( '-p', '--plotResults', nargs='?', metavar='GraphType', choices=('BAG','ERG'), help='plot results obtained from Dijkstra & DynDijkstra Alg. on graph with types: {%(choices)s}')

def main():
    args = parser.parse_args()

    genGraphs = args.genGraphs
    runDijkstra = args.runDijkstra
    plotResults = args.plotResults

    import time
    start = time.process_time()
    if(genGraphs != None):
        import genGraph
        if genGraphs == 'BAG':
            genGraph.clearFolderByType(utility.GraphTypes.BAG)
            genGraph.genGraphByType(utility.GraphTypes.BAG)
        elif genGraphs == 'ERG':
            genGraph.clearFolderByType(utility.GraphTypes.ERG)
            genGraph.genGraphByType(utility.GraphTypes.ERG)
        print(f"{genGraphs} Graphs generated")
    elif(runDijkstra != None):
        import testDijkstra
        if runDijkstra == 'BAG':
            testDijkstra.test_DijkstraOnGraphByType(utility.GraphTypes.BAG)
        elif runDijkstra == 'ERG':
            testDijkstra.test_DijkstraOnGraphByType(utility.GraphTypes.ERG)
        print(f"{genGraphs} Dijkstra computed")
    elif(plotResults != None):
        import plotResult
        if plotResults == 'BAG':
            plotResult.plotGraphByType(utility.GraphTypes.BAG)
        elif plotResults == 'ERG':
            plotResult.plotGraphByType(utility.GraphTypes.ERG)
        print(f"{genGraphs} Plot ended")
    print(f"test ended in {time.process_time() - start} seconds")
    


if __name__ == '__main__':

    main()
    
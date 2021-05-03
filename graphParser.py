import networkit, pandas, sys, logging, os, utility

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("graphParser")
logger.setLevel(logging.DEBUG)

class file_Parser:
    BAG_folder = utility.BAGs_FOLDER # "BarabasiAlbertGraphs/"
    ERG_folder = utility.ERGs_FOLDER # "ErdosRenyiGraphs/"

    BAGs_file_path = []
    BAGs_missing_edges_file_path = []
    ERGs_file_path = []
    ERGs_missing_edges_file_path = []



    def __init__(self) -> None:
        self.findBAGFiles()
        self.findERGFiles()

        self.BAGs_file_path.sort()
        self.BAGs_missing_edges_file_path.sort()
        
        self.ERGs_file_path.sort()
        self.ERGs_missing_edges_file_path.sort()

        self._ERG_iter = 0
        self._BAG_iter = 0

    def findBAGFiles(self):
        for root, dirs, files in os.walk(self.BAG_folder):
            for file in files:
                if(file.__contains__("missingEdgeForBAG")):
                    self.BAGs_missing_edges_file_path.append(os.path.join(root, file))
                elif(file.__contains__(".TabOne")):
                    self.BAGs_file_path.append(os.path.join(root, file))

    def findERGFiles(self):
        for root, dirs, files in os.walk(self.ERG_folder):
            for file in files:
                if(file.__contains__("missingEdgeForERG")):
                    self.ERGs_missing_edges_file_path.append(os.path.join(root, file))
                elif(file.__contains__(".TabOne")):
                    self.ERGs_file_path.append(os.path.join(root, file))

    # ritorna una tupla [indice, networkit.graph, missing_edge_list] finche' non terminano i file contenenti i grafi BAG
    def getNextBAG(self):
        if(self._BAG_iter < len(self.BAGs_file_path)):
            index = self._BAG_iter

            if(os.path.isfile(self.BAGs_file_path[index])):
                graph = networkit.readGraph(self.BAGs_file_path[index], networkit.Format.EdgeListTabOne)
            else:
                return "not_exist"

            if(os.path.isfile(self.BAGs_missing_edges_file_path[index])):
                missing_edges = pandas.read_json(self.BAGs_missing_edges_file_path[index])
            else:
                return "not_exist"

            self._BAG_iter += 1
            return [index, graph, missing_edges]

        else:
            return "no_more_graphs"

    # ritorna una tupla [indice, networkit.graph, missing_edge_list] finche' non terminano i file contenenti i grafi BAG
    def getNextERG(self):
        if(self._ERG_iter < len(self.ERGs_file_path)):
            index = self._ERG_iter

            if(os.path.isfile(self.ERGs_file_path[index])):
                graph = networkit.readGraph(self.ERGs_file_path[index], networkit.Format.EdgeListTabOne)
            else:
                return "not_exist"

            if(os.path.isfile(self.ERGs_missing_edges_file_path[index])):
                missing_edges = pandas.read_json(self.ERGs_missing_edges_file_path[index])
            else:
                return "not_exist"

            self._ERG_iter += 1
            return [index, graph, missing_edges]

        else:
            return "no_more_graphs"

    def resetBAGIterator(self):
        self._BAG_iter = 0

    def resetERGIterator(self):
        self._ERG_iter = 0



def test_getNextBAG():
    parser = file_Parser()
    
    while(True):
        response = parser.getNextBAG()

        if(response == "no_more_graphs" or response == "not_exist"):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"break: {response}")
            break
        else:
            _index = response[2].index

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"test_getNextBAG: {response[0]}")

def test_getNextERG():
    parser = file_Parser()
    
    while(True):
        response = parser.getNextERG()

        if(response == "no_more_graphs" or response == "not_exist"):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"break: {response}")
            break
        else:
            _index =  response[2].index
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"index: {response[0]}, len(){len(response[2].index)}")

if __name__ == "__main__":
    # parser = file_Parser()
    # test_getNextBAG()
    # test_getNextERG()
    pass





    
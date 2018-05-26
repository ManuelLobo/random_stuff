
class routes:
    def __init__(self):
        self.nodes = []



class node:

    def __init__(self, node):
        self.node = node
        self.output_nodes = {}

    def add_output_node(self, node, distance):
        self.output_nodes[node] = distance



def process_file(nodes_file):
    number_of_nodes = int(next(nodes_file))
    nodes_list = set()
    node_dictionary = {}  # tupple : distance

    for i in range(number_of_nodes):
        nodes_list.add(next(nodes_file))

    number_of_edges = int(next(nodes_file))
    print(number_of_edges)

    for i in range(number_of_edges):
        line = next(nodes_file).strip().split(" ")
        node1 = line[0]
        node2 = line[1]
        distance = line[2]
        node_dictionary[(node1, node2)] = distance

    return nodes_list, node_dictionary

#def search_shortest_path():




if __name__ == '__main__':
    nodes_file = open("graph.dat")
    node_list, node_dictionary = process_file(nodes_file)

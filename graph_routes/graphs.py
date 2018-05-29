"""Routes a file of path and finds shortest path."""


class Route:
    def __init__(self):
        self.nodes = []
        self.name_node_dictionary = {}  # code string to node object
        self.best = 99999

    def add_node(self, node):
        self.nodes.append(node)
        self.name_node_dictionary[node.code] = node

    def find_shortest_route(self, start_node, end_node, routes_passed=[], total_distance=0):
        routes_passed = routes_passed + [start_node]
        if start_node == end_node:
            return routes_passed
        if len(routes_passed) > self.best:
            return None
        start_node_object = self.name_node_dictionary[start_node]
        shortest_route = None
        for output_node in start_node_object.output_nodes:
            if output_node not in routes_passed:
                route = self.find_shortest_route(output_node, end_node, routes_passed, total_distance)
                if route:
                    if not shortest_route or len(route) < len(shortest_route):
                        shortest_route = route
                        self.best = len(route)
        return shortest_route

class Node:
    def __init__(self, code):
        self.code = code
        self.output_nodes = {}

    def add_output_node(self, node, distance):
        self.output_nodes[node] = distance

    def __eq__(self, node):
        self.code == node.code


def prepare_data(graph_file):
    """Prepare data from file to data strutures.

    Args:
        nodes_file (str): path to file containing graph information.

    Returns:
        dict: a dictionary of node to output node and distance.
    """
    nodes_file = open(graph_file)
    number_of_nodes = int(next(nodes_file))
    node_list = set()

    for i in range(number_of_nodes):
        node_entry = next(nodes_file).strip()
        node_list.add(node_entry)

    number_of_edges = int(next(nodes_file))

    node_dictionary = {node: [] for node in node_list}
    for i in range(number_of_edges):
        line = next(nodes_file).strip().split(" ")
        node1 = line[0]
        node2 = line[1]
        distance = int(line[2])
        node_dictionary[node1].append((node2, distance))
        node_dictionary[node2].append((node1, distance))

    return node_dictionary


def prepare_routes(node_dictionary):
    route = Route()
    for item in node_dictionary:
        node = Node(item)
        for tup in node_dictionary[item]:
            output_node = tup[0]
            distance = tup[1]
            node.add_output_node(output_node, distance)
        route.add_node(node)
        route.name_node_dictionary[node.code] = node

    return route



if __name__ == '__main__':
    node_dictionary = prepare_data("graph.dat")
    route = prepare_routes(node_dictionary)
    #route.find_shortest_route("316319897", "316319936")
    a = route.find_shortest_route("314248302", "316320382")
    print(a)

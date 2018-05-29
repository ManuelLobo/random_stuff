from graph import Route, Node

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

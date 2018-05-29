

class Route:
    """Structure containing Node objects to find the shortest route.

    The default "best" variable value is used to avoid maximum recursion or
    just a high running time.
    """

    def __init__(self):
        self.nodes = []
        self.name_node_dictionary = {}  # code string to node object
        self.best = 5000  # stores the best (shortest) distance
        self.shortest_route = None

    def add_node(self, node):
        """Add a node to the Route object.

        Add the node to the list of nodes and add a mapping between the node
        name and the node object.

        Args:
            node (Node): a Node object.
        """
        self.nodes.append(node)
        self.name_node_dictionary[node.code] = node

    def find_shortest_route(self, start_node, end_node, routes_passed=[],
                            total_distance=0):
        """Find the shortest route from the start node to the end node.

        Args:
            start_node (str): the starting node.
            end_node (str): the last node.
            routes_passed (list): the travelled nodes.
            total_distance (int): the cumulative distance travelled.

        Returns:
            list: the route with the smallest distance from the start node to
                  the end node.
        """
        routes_passed = routes_passed + [start_node]

        if start_node == end_node:
            self.best = self.get_route_distance(routes_passed)
            return routes_passed
        if total_distance > self.best:
            return None

        start_node_object = self.name_node_dictionary[start_node]
        shortest_route = None
        for output_node in start_node_object.output_nodes:
            if output_node not in routes_passed:
                distance = start_node_object.output_nodes[output_node]
                total_distance = total_distance + distance
                route = self.find_shortest_route(output_node, end_node,
                                                 routes_passed, total_distance)

                if route:
                    if not shortest_route or total_distance < self.best:
                        shortest_route = route
                        self.shortest_route = shortest_route

        return shortest_route

    def get_route_distance(self, route):
        """Find the distance between for a full route.

        Args:
            route (list): list of node strings.

        Returns:
            int: the distance in meters for the full route.
        """
        if route is not None:
            total_distance = 0
            for i in range(len(route) - 1):
                node_object = self.name_node_dictionary[route[i]]
                distance = node_object.get_distance(route[i+1])
                total_distance += distance
            return total_distance
        else:
            return "No route was found."


class Node:
    """Structure containing information on output nodes and distances.

    Args:
        code (str): string representing a node.
    """
    def __init__(self, code):
        self.code = code
        self.output_nodes = {}  # dictionary that maps output node to distance

    def add_output_node(self, output_node, distance):
        """Add an output node to current Node object.

        Args:
            output_node (str): string representation of the output node.
            distance (int): distance in meters from node to output node.

        """
        self.output_nodes[output_node] = distance

    def get_distance(self, output_node):
        """Get distance from the current node to an output node.

        Args:
            output_node (str): string representation of the output node.
        """
        return self.output_nodes[output_node]

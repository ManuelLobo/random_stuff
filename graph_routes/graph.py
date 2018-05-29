class Route:
    def __init__(self):
        self.nodes = []
        self.name_node_dictionary = {}  # code string to node object
        self.best = 5000  # arbitrary maximum total distance
        self.shortest_route = None

    def add_node(self, node):
        self.nodes.append(node)
        self.name_node_dictionary[node.code] = node

    def get_route_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            node_object = self.name_node_dictionary[route[i]]
            distance = node_object.get_distance(route[i+1])
            total_distance += distance
        return total_distance

    def find_shortest_route(self, start_node, end_node, routes_passed=[],
                            total_distance=0):
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


class Node:
    def __init__(self, code):
        self.code = code
        self.output_nodes = {}

    def add_output_node(self, node, distance):
        self.output_nodes[node] = distance

    def get_distance(self, output_node):
        return self.output_nodes[output_node]

    def __eq__(self, node):
        self.code == node.code

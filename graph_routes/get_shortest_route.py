from prepare import prepare_data, prepare_routes
import argparse

def main():
	parser = argparse.ArgumentParser(description='Shortest Route')
	parser.add_argument('-g', '--graph', help='input graph file" ', type=str,
					    dest="graph", required=True)
	parser.add_argument('-s', '--start', help='start node" ', type=str,
						dest="start", required=True)
	parser.add_argument('-f', '--finish', help='finish node" ', type=str,
						dest="finish", required=True)
	args = parser.parse_args()

	node_dictionary = prepare_data(args.graph)
	route = prepare_routes(node_dictionary)
	route.find_shortest_route(args.start, args.finish)
	print(route.get_route_distance(route.shortest_route))


if __name__ == '__main__':
	main()

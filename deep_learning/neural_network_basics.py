import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        # for every node in the input, we want to append the particular
        # operation to the list of output nodes

        for node in input_nodes: # for every node from which the "self" receives an input
            node.output_nodes.append(self) # add itself as a output node to that input node

        _default_graph.operations.append(self)

        def compute(self):
            pass


class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])  # [x, y] is a list of input nodes


    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])  # [x, y] is a list of input nodes


    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matrix_multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])  # [x, y] is a list of input nodes


    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)




class Placeholder():

    def __init__(self):
        self.output_nodes = []

        _default_graph.placeholders.append(self)



class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)


class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


def traverse_postorder(operation):
    """Makes sure computations are done in the correct order.

    e.g. z = Ax + b -> Ax is done first, and then Ax + b
    """

    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
            nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder

class Session():
    """The feed dictionary is a dictionary mapping placeholders to input values."""
    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            print(type(node))
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                print(node.input_nodes)
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.ouput)

        return operation.output


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])

    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))


if __name__ == '__main__':
    #z = Ax + b

    # Create graph and set global default
    g = Graph()
    g.set_as_default()

    # Set variables
    A = Variable(10)
    b = Variable(1)
    x = Placeholder()
    y = multiply(A, x)

    # compute z
    z = add(y, b)

    # Use PostOrder Tree Traversal to execute nodes in right order.

    sess = Session()
    result = sess.run(operation=z, feed_dict={x: 10})

#    sample_z = np.linspace(-10, 10, 100)
    #sample_a = sigmoid(sample_z)
    #plt.plot(sample_z, sample_a)

    # data = tuplue of values and labels
    data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
    features = data[0]
    labels = data[1]
    plt.scatter(features[:, 0], features[:, 1])


    # Separate blobs

    #np.array([1, 1]).dot(np.array([8], [10])) - 5 # result shows it belongs to one class
    #np.array([1, 1]).dot(np.array([2], [-10])) - 5 # result shows it belongs to the other class


    # Creating a simple neural net for seperating classes
    g = Graph()
    g.set_as_default()
    x = Placeholder()
    w = Variable([1,1])
    b = Variable(-5)
    z = add(matrix_multiply(w, x), b)
    a = Sigmoid(z)
    sess = Session()
    sess.run(operation=a, feed_dict=x: [8, 10]})



    #

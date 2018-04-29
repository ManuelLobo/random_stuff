import numpy
import logging

def get_derivative(func, x):
    """Compute the derivative of func at location x"""
    h = 0.0001 # Step size
    return(func(x+h) - func(x)) / h


def derivative_test():

    def f(x): return x**2

    x = 3
    computed = get_derivative(f, x)
    actual = 2*x

    logging.info(f"{computed}, {actual}")

def get_integral(func, a, b):
    """Compute the area under func between a and b"""
    h = 0.0001
    x = a
    total = 0
    while x <= b:
        total += h*func(x)
        x += h
    return total

def integral_test():
    def f(x): return x**2
    computed = get_integral(f, 1, 3)
    def actualF(x): return 1.0/3.0*x**3
    actual = actualF(3)
    logging.info(f"{computed}, {actual}")


def main():
    logging.basicConfig(level=logging.DEBUG,
      format="[%(asctime)s] [%(processName)s:%(threadName)s] "
             "[test-project/%(name)s.%(funcName)s:%(lineno)d] "
             "[%(levelname)s] %(message)s")


    logging.info("Test - Calculate Derivative:")
    derivative_test()
    integral_test()






if __name__ == '__main__':
    main()

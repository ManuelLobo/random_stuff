import argparse

parser = argparse.ArgumentParser(description="Description")
parser.add_argument("-i", type=str, help="Help", required=True)
parser.add_argument("-o", type=str, help="help", required=False)


args = parser.parse_args()

i = args.i
print(i)

if args.o:
    print(args.o)

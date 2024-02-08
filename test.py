import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load', dest='load', action='store_true')

parser.set_defaults(feature=False)

load = parser.parse_args().load
print('Load: ', load)
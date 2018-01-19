# from yaml import load, dump
# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper

import yaml

stream = file('scenario/scenario.yaml', 'r')

data = yaml.load(stream)
data = data['scenario']
initial_conditions = data['InitialConditions']

print len(initial_conditions['TLE_target'])

i = float(initial_conditions['TLE_target'][8:16])

print i
print initial_conditions['TLE_target'][8:16]
print initial_conditions['TLE_target'][17:25]
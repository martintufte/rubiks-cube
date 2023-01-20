import sys
from OLLs import OLLs
from utilities import permutations, invert_seq, get_cube_state

oll = OLLs[sys.argv[1]]
inv_oll = invert_seq(oll)

cube_state = get_cube_state('OLL')

for move in inv_oll.split():
  cube_state = [cube_state[i] for i in permutations[move]]

# set the cube state
print('\\xdef\\myarray{{"' + '","'.join(cube_state) + '"}}')

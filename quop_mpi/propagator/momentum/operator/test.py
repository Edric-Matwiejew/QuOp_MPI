import sys
sys.path.append('../../../../')
import grids

eigen = grids.composite(Ns = (10,5,6), Cs = (2,2,2))

for i in range(3):
    print(eigen[:,i])

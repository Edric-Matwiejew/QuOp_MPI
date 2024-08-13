import sys
sys.path.insert(0, '../')
import os
from glob import glob
from mpi4py import MPI
import numpy as np
from quop_mpi.__utils.__tracker import swarm_tracker
from quop_mpi.__utils.__mpi import subcomms
from time import sleep
from inspect import getmembers

MPI_COMM = MPI.COMM_WORLD

s = subcomms(1, 4, 1, MPI_COMM)

tasks = [i for i in range(29)]

t = swarm_tracker(tasks, 1, s, suspend_path = 'suspend')
print(t.complete)
while not t.complete:
    #print(t.local_tasks)
    #print(t.get_task(), MPI_COMM.Get_rank(), flush = True)
    task = t.get_task()
    #if MPI_COMM.Get_rank() == 0:
    sleep(0.1)
    t.update([task])

    #print(t.get_results())
print(t.get_results())


#class dummy_job:
#    def __init__(self, repeats, depths, time_limit, MPI_COMM):
#
#        suspend_file = "suspend.quop"
#
#        tracker = job_tracker(self, repeats, depths, time_limit, MPI_COMM, suspend_path = "suspend_path")
#
#        print(tracker.get_job(), flush = True)
#        while not tracker.complete:
#            repeat, depth = tracker.get_job()
#            seed = tracker.get_seed()
#            #if repeat == 0:
#            #    if MPI_COMM.Get_rank() == 0:
#            #        print(f'setup for depth {depth}.')
#            results = tracker.get_results()
#            if MPI_COMM.Get_rank() == 0:
#                print(f'repeat {repeat}, depth {depth}, seed {seed}.')
#                print(results)
#            sleep(0.2)
#            tracker.update({'hi':'hello'})
#        
#
#        if MPI_COMM.Get_rank() == 0:
#            print(tracker.results_dict, flush = True)
#            if tracker.complete:
#                suspend_files = glob("*.quop")
#                for sfile in suspend_files:
#                    os.remove(sfile)
#
#
#repeats = 5
#depths = 8
#time_limit = 1.5
#
#dummy_job(repeats, depths, time_limit, MPI_COMM)

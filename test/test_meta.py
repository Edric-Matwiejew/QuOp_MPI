import meta
from quop_mpi import Ansatz

COMM = meta.MPI.COMM_WORLD

class test_swarm(meta.swarm):
    @meta.iterate_subcomms_1
    def set_method_1(self, arg1, arg2, key = None):
        print('1', arg1, arg2, key, flush = True)
    @meta.iterate_subcomms_1
    def set_method_2(self, key_1 = None, key_2 = None):
        print('2', key_1, key_2, flush = True)

def test_arg_pass():

    test = test_swarm(1, 4, 4, COMM)

    test.set_unitaries(['one', 'two'])
    test.set_unitaries(4*[['one', 'two']])
    #test.set_unitaries(3*[['one', 'two']])
    test.set_method_1(
            [
                ['one', 'two', {'key':1}],
                ['one', 'two', {'key':1}],
                ['one', 'two', {'key':1}],
                ['one', 'two', {'key':1}]
                ]
                )

    #            #['three', 'four'],
    #            #['five', 'six'],
    #            #['seven', 'eight']
    #            #{'key':1},
    #            #{'key':2},
    #            #{'key':3},
    #            #]
    #        #)

    COMM.barrier()
    test.set_method_1('one', 'two', key = 1)
    COMM.barrier()
    test.set_method_2(key_1 = 1)
    COMM.barrier()
    test.set_method_1('one', 'two')
    #COMM.barrier()
    #try:
    #    test.set_method_1(['one', 'two'])
    #except Exception as e:
    #    print(e)
    #COMM.barrier()
    #test.set_method_1(['one one', 'one two'])
    #test.set_method_1(4*[['one', 'two']], key = 1)
    #COMM.barrier()
    #try:
    #    test.set_method_1(4*['one'], key = 1)
    #except Exception as e:
    #    print(e)
    #size = test.get_size()
    #test.set_method_1(size*[['arg']], size*[{'key':'kwarg'}])

def test_is_root():
    test = test_swarm(1, 4, 2, COMM)
    print(test.is_root())

def test_subcomms():
    test = test_swarm(1, 4, 2, COMM)
    if COMM.Get_rank() == 0:
        print(test.get_groups())

    test = test_swarm(1, 4, 4, COMM)
    if COMM.Get_rank() == 0:
        print(test.get_groups())

    test = test_swarm(1, 3, 2, COMM)
    if COMM.Get_rank() == 0:
        print(test.get_groups())


    test = test_swarm(2, 4, 2, COMM)
    if COMM.Get_rank() == 0:
        print(test.get_groups())


def test_wrap():
    from quop_mpi import Ansatz
    test = meta.swarm(1, 1, 1, COMM)
    test.set_ansatz(Ansatz)


if __name__ == "__main__":
    test_wrap()
    #test_arg_pass()
    #COMM.barrier()
    #test_is_root()
    #COMM.barrier()
    #test_subcomms()

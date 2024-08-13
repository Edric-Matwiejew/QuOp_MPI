from glob import glob
from shutil import rmtree
from time import localtime, mktime, asctime
from mpi4py import MPI
from quop_mpi.__utils import __filenames

def test_ensure_path_and_extension():
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("testing filename modification...")
        filepath = "test"
        print(__filenames.ensure_path_and_extension(filepath, "dummy", "modifier", ensure_path = False)) 
        filepath = "test.dummy"
        print(__filenames.ensure_path_and_extension(filepath, "dummy", "modifier", ensure_path = False))
        filepath = "test.test.dummy"
        print(__filenames.ensure_path_and_extension(filepath, "dummy", "modifier", ensure_path = False)) 
        filepath = "//test.test/test.test.dummy"
        print(__filenames.ensure_path_and_extension(filepath, "dummy", "modifier", ensure_path = False)) 
        filepath = "//test.test.dummy/test.test.dummy"
        print(__filenames.ensure_path_and_extension(filepath, "dummy", "modifier", ensure_path = False)) 
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("\ntesting filepath creation...")
        rmtree("test")
    filepath = "test/test"
    if MPI.COMM_WORLD.Get_rank() == 0:
        filepath = __filenames.ensure_path_and_extension(filepath, "dummy", ensure_path = True)
        with open(filepath, "w") as f:
            f.write("dummy data")
    if MPI.COMM_WORLD.Get_rank() == 0:
        filepath = __filenames.ensure_path_and_extension(filepath, "dummy", unique = True, ensure_path = True)
        with open(filepath, "a") as f:
            f.write("dummy data")
        for path in glob("test/*"):
            print(path)
        print("\ntesting MPI filepath creation...")
        rmtree("test")
    filepath = "test/test"
    filepath = __filenames.ensure_path_and_extension(filepath, "dummy", unique = True, MPI_COMM = MPI.COMM_WORLD, ensure_path = True, add_time = True)
    with open(filepath, "a") as f:
        f.write("dummy data")
    if MPI.COMM_WORLD.Get_rank() == 0:
        for path in glob("test/*"):
            print(path)

test_ensure_path_and_extension()


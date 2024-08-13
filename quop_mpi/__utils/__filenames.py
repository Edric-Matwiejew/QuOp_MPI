from os import path
from pathlib import Path
from time import localtime, asctime


def time_suffix():
    time_string = asctime(localtime())
    return "_".join(("_".join(time_string.split(" "))).split(":"))


def ensure_directory(filepath):
    Path(path.dirname(filepath)).mkdir(parents=True, exist_ok=True)


def ensure_uniqueness(filepath, extension, MPI_COMM, add_time):
    filepath = (
        modify_file_basename(filepath, time_suffix(), extension)
        if path.exists(filepath) or add_time
        else filepath
    )

    if MPI_COMM is not None:

        filepath = MPI_COMM.bcast(filepath, root=0)

        filepath = modify_file_basename(filepath, MPI_COMM.Get_rank(), extension)

    return filepath


def ensure_extension(filepath, extension):
    if len(filepath) > len(extension) + 1:
        has_extension = filepath[-len(extension) :] == extension
    else:
        has_extension = False
    if not has_extension:
        filepath += f".{extension}"
    return filepath


def ensure_path_and_extension(
    filepath,
    extension,
    modifier=None,
    unique=False,
    MPI_COMM=None,
    ensure_path=True,
    add_time=False,
):
    if ensure_path:
        ensure_directory(filepath)
    filepath = ensure_extension(filepath, extension)
    if modifier is not None:
        filepath = modify_file_basename(filepath, modifier, extension)
    return (
        ensure_uniqueness(filepath, extension, MPI_COMM, add_time)
        if unique
        else filepath
    )


def modify_file_basename(filepath, modifier, extension):
    slist = filepath.split(".")
    slist[-2] = f"{slist[-2]}_{modifier}"
    return ".".join(slist)

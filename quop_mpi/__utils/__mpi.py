import numpy as np
from mpi4py import MPI
from logging import warn

MPI_COMM_type = type(MPI.COMM_WORLD)


def __scatter_1D_array(array, partition_table, MPI_COMM, dtype):

    rank = MPI_COMM.Get_rank()
    local_i = partition_table[rank + 1] - partition_table[rank]
    operator = np.empty(local_i, dtype)

    counts = partition_table[1:] - partition_table[:-1]
    disps = partition_table[:-1] - 1

    if dtype == np.complex128:
        send_type = MPI.DOUBLE_COMPLEX
    elif dtype == np.float64:
        send_type = MPI.DOUBLE

    MPI_COMM.Scatterv([array, counts, disps, send_type], operator[:local_i], 0)

    return operator


def __scatter_sparse(row_starts, col_indexes, values, partition_table, MPI_COMM):

    rank = MPI_COMM.Get_rank()
    size = MPI_COMM.Get_size()

    lb = partition_table[rank] - 1
    ub = partition_table[rank + 1] - 1

    if rank == 0:
        n_terms = MPI_COMM.bcast(len(row_starts), 0)
    else:
        n_terms = MPI_COMM.bcast(None, 0)

    W_row_starts = []
    W_col_indexes = []
    W_values = []

    for i in range(n_terms):

        n_local_rows = partition_table[rank + 1] - partition_table[rank]

        W_row_starts.append(np.empty(n_local_rows + 1, dtype = np.int64))

        counts = partition_table[1:] - partition_table[0:-1] + 1
        disps = partition_table[:-1] - 1

        if rank == 0:
            sends = [row_starts[i].astype(np.int64), counts, disps, MPI.LONG]
        else:
            sends = None  # [None, counts, disps, MPI.INT]

        MPI_COMM.Scatterv(sends, W_row_starts[-1], 0)

        n_local_nnz = W_row_starts[-1][-1] - W_row_starts[-1][0]

        W_col_indexes.append(np.empty(n_local_nnz, dtype = np.int64))
        W_values.append(np.empty(n_local_nnz, np.complex128))

        counts = np.zeros(size, int)
        counts[rank] = n_local_nnz

        for j in range(size):
            counts[j] = MPI_COMM.bcast(n_local_nnz, j)

        disps = [0 for _ in range(size)]
        for j in range(1, size):
            disps[j] = disps[j - 1] + counts[j - 1]

        if rank == 0:
            send_indexes = [col_indexes[i].astype(np.int64), counts, disps, MPI.LONG]
            send_values = [
                values[i].astype(np.complex128),
                counts,
                disps,
                MPI.DOUBLE_COMPLEX,
            ]
        else:
            send_indexes = None  # [None, counts, disps, MPI.INT]
            send_values = None  # [None, counts, disps, MPI.DOUBLE_COMPLEX]

        MPI_COMM.Scatterv(send_indexes, W_col_indexes[-1], 0)
        MPI_COMM.Scatterv(send_values, W_values[-1], 0)

    return W_row_starts, W_col_indexes, W_values


def shrink_communicator(newsize, colours, COMM, COMM_OPT, COMM_JAC, jac_ranks):

    if jac_ranks is not None:
        if COMM.Get_rank() in jac_ranks:
            MPI.Comm.Free(COMM_JAC)

    if colours[COMM.Get_rank()] != -1:

        subcolours = []
        for i in range(COMM_OPT.Get_size()):
            if i < newsize:
                subcolours.append(0)
            else:
                if COMM_OPT.Get_rank() == i:
                    colours[COMM.Get_rank()] = -1
                subcolours.append(MPI.UNDEFINED)

        COMM_OPT_NEW = MPI.Comm.Split(
            COMM_OPT, subcolours[COMM_OPT.Get_rank()], COMM_OPT.Get_rank()
        )
    else:
        COMM_OPT_NEW = None

    for i, colour in enumerate(colours):
        colours[i] = COMM.bcast(colours[i], i)

    if jac_ranks is not None:
        jac_ranks = []
        jac_ranks = [
            rank
            for rank in range(COMM.Get_size())
            if (colours[rank] != 0) and (colours[rank] != -1)
        ]
        jac_ranks.insert(0, 0)

        world_group = MPI.Comm.Get_group(COMM)
        jac_group = MPI.Group.Incl(world_group, jac_ranks)
        COMM_JAC = COMM.Create_group(jac_group)

    return colours, COMM_OPT_NEW, COMM_JAC, jac_ranks


def gather_array(array, partition_table, COMM_OPT):

    if COMM_OPT.Get_rank() == 0:
        gathered_array = np.empty(partition_table[-1] - 1, type(array[0]))
    else:
        gathered_array = None

    counts = partition_table[1:] - partition_table[:-1]

    COMM_OPT.Gatherv(array, [gathered_array, counts], 0)

    return gathered_array


class subcomms:
    def __init__(
        self,
        nodes_per_subcomm,
        processes_per_node,
        maxcomm,
        MPI_COMM,
    ):

        """
        nodes_per_subcomm: The number of nodes in each subcommunicator. 
        processes_per_node: The number of processes at each node.
        macomm: Try to create this many subcommunicators.

        This will create 4 subcommunicators with 16 processors each.
        Requested nodes: 4
        nodes_per_subcomm = 1
        processes_per_node = 16
        maxcomm = 4

        This will create 2 subcommuniators with 32 processors each.
        Requested nodes: 4
        nodes_per_subcomm = 2
        processes_per_node = 16
        maxcomm = 2

        This will create 8 subcommunicators with 16 processes each.
        Requested nodes: 4
        nodes_per_subcomm = 1
        processes_per_node = 16
        maxcomm = 8

        This will create 4 subcommunicators with 4 processes each.
        Requested nodes: 1
        nodes_per_subcomm = 1
        processes_per_node = 16
        maxcomm = 4
        """

        self.MPI_COMM = MPI_COMM

        if nodes_per_subcomm is None:

            self.groups = [[i for i in range(self.MPI_COMM.Get_size())]]
            self.colour = 0

        else:


            node_ID = [MPI.Get_processor_name(), self.MPI_COMM.Get_rank()]

            IDs = self.MPI_COMM.allgather(node_ID)

            if self.MPI_COMM.Get_rank() == 0:

                self.nodes_dict = {}
                for ID in IDs:
                    self.nodes_dict[ID[0]] = []

                for ID in IDs:
                    self.nodes_dict[ID[0]].append(ID[1])

                for node in self.nodes_dict.keys():

                    if len(self.nodes_dict[node]) < processes_per_node:
                        n_processes = len(self.nodes_dict[node])
                        warn(
                            (
                                f"Number of requested processes at node {node} "
                                f"is less than {processes_per_node}, using "
                                f"{n_processes} instead."
                            )
                        )
                    else:
                        n_processes = processes_per_node

                    self.nodes_dict[node] = self.nodes_dict[node][:n_processes]

                nodes = list(self.nodes_dict.keys())
                n_nodes = len(nodes)

                if nodes_per_subcomm == 1:

                    
                    if n_processes % maxcomm != 0:
                        n_subcomms = max([1, n_processes // maxcomm])
                        warn(
                            (
                                f"Cannot create {maxcomm} equal-sized "
                                f"subcommunicators on 1 physical node with "
                                f"{n_processes} total processes, creating "
                                f"{n_subcomms} instead."
                            )
                        )
                    else:
                        n_subcomms = maxcomm

                    groups = []
                    for node in nodes:
                        groups += np.split(np.array(self.nodes_dict[node]), n_subcomms)

                else:

                    if n_nodes % nodes_per_subcomm != 0:
                        n_subcomms = max([1, n_nodes // nodes_per_subcomm])
                        warn(
                            (
                                f"Cannot create {maxcomm} equal-sized "
                                f"subcommunicators with {n_nodes} total "
                                f"physical nodes with {nodes_per_subcomm} "
                                f"nodes per subcommunicators, creating {n_subcomms} "
                                f"instead."
                            )
                        )
                    else:

                        n_subcomms = maxcomm

                    subcomm_nodes = np.split(np.array(nodes), n_subcomms)

                    groups = []

                    for subcomm_node_group in subcomm_nodes:
                        groups.append([])
                        for node in subcomm_node_group:
                            groups[-1] += self.nodes_dict[node]

                        groups[-1] = np.array(groups[-1], dtype=int)

            else:

                groups = None

            self.groups = self.MPI_COMM.bcast(groups, root=0)

            for i, group in enumerate(self.groups):
                if MPI_COMM.Get_rank() in group:
                    self.colour = i
                    break
                self.colour = -1

        if len(self.groups) > 1:

            for group in self.groups:

                if self.MPI_COMM.Get_rank() in group:

                    GROUP = self.MPI_COMM.group.Incl(group)
                    self.SUBCOMM = MPI.Intracomm.Create_group(self.MPI_COMM, GROUP)
        else:

            self.SUBCOMM = self.MPI_COMM

        self.roots = []
        for group in self.groups:
            self.roots.append(group[0])

        self.WORLDGROUP = self.MPI_COMM.Get_group()

        ROOTGROUP = self.WORLDGROUP.Incl(self.roots)

        self.ROOTCOMM = MPI.Intracomm.Create_group(self.MPI_COMM, ROOTGROUP)

        self.size = len(self.groups)

        self.jaccomm_group = None
        self.JACCOMM = None

    def create_jaccomm(self):

        if self.get_n_subcomms() == 1:
            warn('One subcommunicator present, skipping creation of JACOMM.')

        else:

            group_indexes = range(1, self.get_n_subcomms())

            self.jaccomm_group = [
                self.groups[index][i]
                for index in group_indexes
                for i in range(len(self.groups[index]))
            ]

            self.jaccomm_group.insert(0, 0)

            if self.MPI_COMM.Get_rank() in self.jaccomm_group:
                GROUP = self.MPI_COMM.group.Incl(self.jaccomm_group)
                self.JACCOMM = MPI.Intracomm.Create_group(self.MPI_COMM, GROUP)

    def get_n_subcomms(self):
        return len(self.groups)

    def get_subcomm_groups(self):
        return self.groups

    def get_jaccomm_group(self):
        return self.jaccomm_group

    def get_subcomm_index(self):
        return self.colour

    def get_subcomm_roots(self):
        return self.roots

    def in_rootcomm(self):
        return self.MPI_COMM.Get_rank() == self.roots[self.colour]

    def in_subcomm(self):
        return self.colour >= 0

    def in_jaccomm(self):
        return not self.JACCOMM is None

    def count_empty_subcomm_ranks(self, local_i):

        empty = None
        n_empty_in_subcomm = None

        if self.in_subcomm():
            empty = 0 if local_i > 0 else 1
            n_empty_in_subcomm = self.SUBCOMM.allreduce(empty, op=MPI.SUM)

        return n_empty_in_subcomm


    def shrink_subcomms(self, n_empty_in_subcomm):

        if self.in_subcomm():

            drop_index = len(self.groups[self.get_subcomm_index()]) - n_empty_in_subcomm
            self.groups[self.get_subcomm_index()] = self.groups[
                self.get_subcomm_index()
            ][0:drop_index]

            self.free_subcomm()

            if self.MPI_COMM.Get_rank() in self.groups[self.get_subcomm_index()]:
                SUBCOMMGROUP = self.WORLDGROUP.Incl(
                    self.groups[self.get_subcomm_index()]
                )
                self.SUBCOMM = MPI.Intracomm.Create_group(self.MPI_COMM, SUBCOMMGROUP)
            else:
                self.colour = -1

    def free_rootcomm(self):
        if self.in_rootcomm():
            MPI.Comm.Free(self.ROOTCOMM)

    def free_subcomm(self):
        if self.in_subcomm():
            if self.SUBCOMM != self.MPI_COMM:
                MPI.Comm.Free(self.SUBCOMM)

    def free_jaccomm(self):
        if not self.JACCOMM is None:
            MPI.Comm.Free(self.JACCOMM)

    def free(self):
        self.free_jaccomm()
        self.free_rootcomm()
        self.free_subcomm()

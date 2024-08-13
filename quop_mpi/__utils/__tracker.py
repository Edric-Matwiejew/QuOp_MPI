from os import path, environ
from glob import glob
import sys
import pickle
import inspect
from time import time
from copy import copy
from logging import warn, info
import numpy as np
from mpi4py import MPI
from quop_mpi.__utils.__filenames import ensure_path_and_extension
import __main__


def root_print(MPI_COMM, message):
    if MPI_COMM.Get_rank() == 0:
        print(message, flush=True)


class swarm_tracker:
    """
    Takes a list of tasks and returns those values when requested.
    Time is marked on request of a task.
    Time is left is computed when updateded.
    Records which tasks are complete and which result is assocaited with each task.
    Can I use some of atomic operation such that jobs are given to a swarm as needed?
    Each task is associated with a unique seed.
    More jobs can be added to the task list. If there are already existing tasks,
    those that do not match any of the tasks will be carried out.
    The order of the tasks is followed, if the task list is expanded, matching tasks
    are 'copied' into new expanded lists.

    Resume functionality will be less intelligent here.


    tasks = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p] <- divided between swarms
    status = [True, True, False, True, True, ...] <- has the task been complete anywhere?
    seeds = [0, 1, 2, 3, 4, 5...] <- unique seed for each task
    barrier_test <- returns true when the next job passes a test, passed the
    next job and the previous job.

    after_barrier <- executed after the barrier


    saves: tasks, status, seeds, results_dicts

    MPI_COMM.Get_rank() == 0, runs the show.

    results_dict = {task:quop_result, ...}

    checks for incoming messages (tagged by seed, track the next seed),
    if there is a message meeting that tag, recieve the message, update
    status, append to results_dict, dump suspend file.

    alg.do_task(tracker.get_task())
    tracker.update(alg.quop_result())

    def get_task(self):
        returns task and seed

    def update(self):
    if in subcomm
        if self.COMM.Get_rank() == 0:
            check for incomming messages
            add own results to reuslts_dict
            dump
        else:
            non-blocking send to self.COMM.Get)rank() == 0:

    def __dump(self):


    def __distribute_tasks(self):

    def __mark_time(self):


    """

    def __init__(
        self, tasks, time_limit, subcomms, force_resume=None, suspend_path=None, seed=0
    ):
        self.cnt = 0

        self.tasks = tasks
        self.time_limit = time_limit
        self.subcomms = subcomms
        self.force_resume = force_resume
        self.suspend_path = suspend_path
        self.seed = 0

        self.__set_with_environment_variable(
            force_resume, "force_resume", "QUOP_FORCE_RESUME", int
        )

        self.__set_with_environment_variable(
            time_limit, "time_limit", "QUOP_TIME_LIMIT", float
        )
        self.__set_with_environment_variable(
            suspend_path, "suspend_path", "QUOP_SUSPEND_PATH", str
        )

        self.suspend_path = (
            ensure_path_and_extension(suspend_path, "quop")
            if self.suspend_path is not None
            else self.suspend_path
        )

        self.source = self.__get_source()
        self.complete = False
        self.got_match = False
        self.marked_time = None
        self.task_index = 0

        # used by root only
        self.seeds = None
        self.seed_int = self.seed
        self.suspend_dict = {}
        self.seeds_dict = {}
        self.status = None

        self.suspend_signal_tag = None
        self.results_dict = {}

        self.subcomm_status = [
            [True, False] for _ in range(self.subcomms.get_n_subcomms())
        ]

        self.sent_status = False

        self.status_send_tags = [
            len(self.tasks) + self.subcomms.get_n_subcomms()*self.subcomms.get_subcomm_index() + i
            for i in range(self.subcomms.get_n_subcomms())
        ]

        self.status_recv_tags = [
           len(self.tasks) + self.subcomms.get_n_subcomms()*i + self.subcomms.get_subcomm_index()
           for i in range(self.subcomms.get_n_subcomms())
                ]

        if self.suspend_path is not None:
            self.__get_match()

        self.__mark_time()

        if (not self.time_limit is None) and self.got_match:
            self.__resume()
            if not self.complete:
                self.__check_completed_tasks()
        else:
            self.__setup_new()

        if not self.complete:
            self.__distribute_tasks()

    def __set_with_environment_variable(
        self, source_value, attribute, environment_variable, typecast
    ):

        env = environ.get(environment_variable)

        if not (env is None):

            env = typecast(env)

            if self.subcomms.MPI_COMM.Get_rank() == 0:
                info(
                    (
                        f"Setting '{attribute}' to '{env}' from "
                        f"environment variable '{environment_variable}'"
                    )
                )

            setattr(self, attribute, env)

    def __check_completed_tasks(self):

        completed_tasks = []
        for key in list(self.results_dict.keys()):
            if not self.results_dict[key] is None:
                completed_tasks.append(key)

        assigned_seeds = list(self.seeds_dict.keys())

        tasks = []
        seeds = []

        for task, status in zip(self.tasks, self.status):
            if not str(task) in completed_tasks:
                tasks.append(task)
                if task in assigned_seeds:
                    seeds.append(self.seeds_dict[task])
                else:
                    seeds.append(self.seed_int)
                    self.seed_int += 1

        self.tasks = tasks
        self.seeds = seeds

    def __setup_new(self):

        self.results_dict = {str(task): None for task in self.tasks}
        self.seeds = [i + self.seed for i in range(len(self.tasks))]
        self.seeds_dict = {str(task): seed for task, seed in zip(self.seeds, self.tasks)}
        self.seed_int = max(self.seeds) + 1

    def __distribute_tasks(self):
        """
        sends a sublist of the task list to each subcomm.
        generates control flow tags
        """
        # if self.subcomms.MPI_COMM.Get_rank() == 0:

        self.status = [False for i in range(len(self.tasks))]
        self.task_sublists = [None for _ in range(self.subcomms.get_n_subcomms())]
        self.seed_sublists = [None for _ in range(self.subcomms.get_n_subcomms())]
        self.status_sublists = [None for _ in range(self.subcomms.get_n_subcomms())]
        self.confirm_suspends = [False for _ in range(self.subcomms.get_n_subcomms())]

        # check number of tasks, raise error if need.
        for i in range(self.subcomms.get_n_subcomms()):

            self.task_sublists[i] = self.tasks[i :: self.subcomms.get_n_subcomms()]

            self.seed_sublists[i] = self.seeds[i :: self.subcomms.get_n_subcomms()]

            self.status_sublists[i] = self.status[i :: self.subcomms.get_n_subcomms()]

        self.suspend_signal_tags = [
            max(self.seeds) + 1 + i for i in range(self.subcomms.get_n_subcomms())
        ]

        self.suspend_confirmation_tags = [
            self.suspend_signal_tags[-1] + 1 + i
            for i in range(self.subcomms.get_n_subcomms())
        ]

        rank = self.subcomms.get_subcomm_index()
        self.local_tasks = self.task_sublists[rank]
        self.local_seeds = self.seed_sublists[rank]
        self.suspend_signal_tag = self.suspend_signal_tags[rank]
        self.suspend_confirmation_tag = self.suspend_confirmation_tags[rank]
        self.local_results_dict = {str(task): None for task in self.local_tasks}

        # nothing to do - all done!
        if len(self.local_tasks) == 0:
            self.subcomm_status[self.subcomms.get_subcomm_index()] = [True, True]

    def get_task(self):
        """
        returns the next task for a subcomm and mark the time at that subcomm.
        """
        if self.subcomms.in_subcomm():
            self.__mark_time()
            if self.task_index < len(self.local_tasks):
                return self.local_tasks[self.task_index]

    def get_seed(self):
        if self.subcomms.in_subcomm():
            if self.task_index < len(self.local_tasks):
                return self.local_seeds[self.task_index]

    def __collect_results(self, result):

        if self.subcomms.MPI_COMM.Get_rank() == 0:

            # if there's a result at the root rank, add it.
            if not (result is None):
                self.results_dict[str(self.local_tasks[self.task_index])] = result
                self.status_sublists[0][self.task_index] = True

            # looping through all subcomm roots other than the subcomm
            # that contains the rank with global index 0.
            for i, (subcomm_root, tasks, tags) in enumerate(zip(
                self.subcomms.get_subcomm_roots()[1:],
                self.task_sublists[1:],
                self.seed_sublists[1:],
            )
            ):
                for j, (task, tag), in enumerate(zip(tasks, tags,)):
                    if not self.status_sublists[i + 1][j]:
                        if self.subcomms.MPI_COMM.iprobe(source=subcomm_root, tag=tag):
                            self.results_dict[str(task)] = self.subcomms.MPI_COMM.recv(
                                None, source=subcomm_root, tag=tag
                            )
                            self.status_sublists[i + 1][j] = True

            self.status = [status for sublist in self.status_sublists for status in sublist]

        elif self.subcomms.in_rootcomm() and (not result is None):
            self.subcomms.MPI_COMM.isend(
                result, dest=0, tag=self.local_seeds[self.task_index]
            )

    def __enough_time(self):
        if not self.time_limit is None:
            return self.time_limit > 2 * (time() - self.marked_time)
        else:
            return True

    def __update_time(self):

        if not self.time_limit is None:
            self.time_limit -= time() - self.marked_time
            self.time_limit = self.subcomms.SUBCOMM.allreduce(self.time_limit, op=MPI.MIN)

    def __send_status(self, status):
        if self.subcomms.in_rootcomm() and not self.sent_status:
            for rank, tag in enumerate(self.status_send_tags):
                if rank != self.subcomms.ROOTCOMM.Get_rank():
                    self.subcomms.ROOTCOMM.isend(
                        status,
                        dest=rank,
                        tag=tag,
                    )
                    self.sent_status = True

    def __query_status(self):
        if self.subcomms.in_rootcomm():
            for rank, tag in enumerate(self.status_recv_tags):
                if rank != self.subcomms.ROOTCOMM.Get_rank():
                    if self.subcomms.ROOTCOMM.iprobe(source=rank, tag=tag):
                        self.subcomm_status[rank] = self.subcomms.ROOTCOMM.recv(
                            source=rank, tag=tag
                        )

    def __end_tracker(self):
        out_of_time, completed_tasks = self.__end_state()
        return out_of_time or completed_tasks

    def __end_state(self):
        out_of_time = any([not status[0] for status in self.subcomm_status])
        completed_all_tasks = all([status[1] for status in self.subcomm_status])
        return out_of_time, completed_all_tasks

    def __synchronise_tracker(self):
        #exit()
        if self.subcomms.in_rootcomm():
            synchronised = False
            while not synchronised:
                self.__query_status()
                out_of_time, completed_all_tasks = self.__end_state()
                if out_of_time or completed_all_tasks:
                    synchronised = True
            return out_of_time, completed_all_tasks

    def __gather_results(self):
        if self.subcomms.in_rootcomm():
            results = self.subcomms.ROOTCOMM.gather(self.local_results_dict, root=0)
            if self.subcomms.ROOTCOMM.Get_rank() == 0:
                for res in results:
                    for key in res.keys():
                        self.results_dict[key] = res[key]

    def update(self, result):

        if self.subcomms.in_subcomm():

            if not len(self.local_tasks) == self.task_index:
                self.local_results_dict[str(self.local_tasks[self.task_index])] = copy(result)

                self.__collect_results(
                    self.local_results_dict[str(self.local_tasks[self.task_index])]
                )
                self.task_index += 1

                self.subcomms.SUBCOMM.barrier()
                self.__update_time()

                self.subcomm_status[self.subcomms.get_subcomm_index()][
                    1
                ] = self.task_index == len(self.local_tasks)

                self.subcomm_status[self.subcomms.get_subcomm_index()][
                    0
                ] = self.__enough_time()

            status = self.subcomm_status[self.subcomms.get_subcomm_index()]


            if self.subcomms.in_rootcomm():

                if (not status[0]) or status[1]:
                    self.__send_status(status)

                self.__query_status()

                if self.__end_tracker():
                    self.__synchronise_tracker()

            self.subcomms.SUBCOMM.barrier()

            self.subcomm_status = self.subcomms.SUBCOMM.bcast(self.subcomm_status, root=0)

            if self.__end_tracker():
            
                if self.subcomms.MPI_COMM.Get_rank() == 0:
                    self.__collect_results(None)
                out_of_time, completed_all_tasks = self.__end_state()
                self.complete = completed_all_tasks
                self.__dump()
                if out_of_time:
                    self.__suspend()

            if not self.complete:
                return

        self.subcomms.MPI_COMM.barrier()

    def get_results(self):
        """
        gathers and returns all of the results so far to all subcomms.
        """
        #return self.subcomms.MPI_COMM.bcast(self.results_dict, root=0)
        return self.results_dict

    def __mark_time(self):

        if self.subcomms.in_rootcomm():
            self.marked_time = time()
        else:
            self.marked_time = None

        if self.subcomms.in_subcomm():
            self.marked_time = self.subcomms.SUBCOMM.bcast(self.marked_time, 0)

    def __dump(self):

        if (self.subcomms.MPI_COMM.Get_rank() == 0) and (not self.suspend_path is None):
            d = {"source": self.__get_source()}
            backup_attributes = [
                "results_dict",
                "seeds_dict",
                "source",
                "suspend_path",
                "complete",
            ]
            for attribute in backup_attributes:
                d[attribute] = getattr(self, attribute)

            with open(self.suspend_path, "wb") as f:
                pickle.dump(d, f)

    def __suspend(self):

        self.__dump()
        self.subcomms.MPI_COMM.barrier()
        exit(0)

    def __resume(self):
        for key in self.suspend_dict.keys():
            setattr(self, key, self.suspend_dict[key])

        if self.complete:
            root_print(self.subcomms.MPI_COMM, "Job complete.")

    def __get_match(self):

        got_match = False

        if self.subcomms.MPI_COMM.Get_rank() == 0:

            suspend_path = self.suspend_path
            suspend_dict = self.suspend_dict

            exists = path.exists(suspend_path)
            if exists:
                suspend_dict = self.__load_suspened_file(suspend_path)
                got_match, lineold, linenew = self.__is_matching_source(
                    suspend_dict["source"], get_lines=True
                )
                if got_match or (self.force_resume == 1):
                    got_match = True
                    root_print(
                        self.subcomms.MPI_COMM, f"Resuming from '{suspend_path}'."
                    )

            if not got_match:
                suspend_files = glob(f"{path.dirname(suspend_path)}*.quop")

                for sfile in suspend_files:
                    suspend_dict = self.__load_suspened_file(sfile)
                    suspend_path = sfile
                    if self.__is_matching_source(suspend_dict["source"])[0]:
                        got_match = True
                        info(f"Resuming from {sfile}.")
                        break

            if (not got_match) and exists:
                suspend_dict = self.__load_suspened_file(self.suspend_path)
                lines = self.__is_matching_source(
                    suspend_dict["source"], get_lines=True
                )[1:]
                suspend_path = ensure_path_and_extension(
                    self.suspend_path, "quop", add_time=True, unique=True
                )

                warn(
                    (
                        f"Detected changes to '{__main__.__file__}'. "
                        f"\n{lines[0]}\nDoes not match:\n{lines[1]}\n"
                        f"Restarting job and suspending "
                        f"to '{suspend_path}'. Force resume from "
                        f"'{self.suspend_path}' by defining the environment "
                        f"variable: 'QUOP_FORCE_RESUME=1'"
                    )
                )
        else:
            suspend_path = None
            suspend_dict = None

        self.got_match = self.subcomms.MPI_COMM.bcast(got_match, root=0)
        self.suspend_path = self.subcomms.MPI_COMM.bcast(suspend_path, root=0)
        self.suspend_dict = self.subcomms.MPI_COMM.bcast(suspend_dict, root=0)

    def __get_source(self):
        source = inspect.getsource(__main__)
        source = source.splitlines()
        source = [f"Line {i}: {line.strip()}" for i, line in enumerate(source)]
        return "\n".join([s for s in source if s.strip("\r\n").strip()])

    def __is_matching_source(self, source, get_lines=False):

        for current_line, old_line in zip(
            self.source.splitlines(), source.splitlines()
        ):

            if not current_line == old_line:
                if get_lines:
                    return False, current_line, old_line
                else:
                    return False, None, None

        return True, None, None

    def __load_suspened_file(self, path):
        with open(path, "rb") as f:
            suspend_dict = pickle.load(f)
        return suspend_dict


class swarm_benchmark:
    pass


class job_tracker:
    def __init__(
        self,
        repeats,
        max_depths,
        time_limit,
        MPI_COMM,
        force_resume=None,
        suspend_path=None,
        seed=0,
    ):

        """
        Track progression of a benchmark-like job.
        Records the results, allows for a benchmark to be resumed and for additional siulations 
        to be done at a higher depth or more repeats.

        If QUOP_FORCE_RESUME=1 then load from the specified resume file no matter what.
        Otherwise it will only resume if the source matches exactly.

        If set, these environment variables override the corresponding variables in the
        source file.

        export QUOP_REPEATS=5
        export QUOP_MAX_DEPTH=16
        export QUOP_TIME_LIMIT=3600
        export QUOP_SUSPEND_TO=suspend.quop

        By default the suspend files are called 'suspend_<script basename>_<algname>.quop'

        If the source file is modified and QUOP_FORCE_RESUME is not equal to 1 the
        file name is modified with the date and time.

        The class incrementments 'seed' with each call to 'update' and recalls the last
        'seed' value when the job is restored. This way 'seed' can  be used to ensure 
        that additional repeats are unique.

        """
        self.cnt = 0

        self.MPI_COMM = MPI_COMM
        self.seed = seed

        self.depths = range(1, max_depths + 1)
        self.repeats = range(1, repeats + 1)
        self.time_limit = time_limit
        self.suspend_path = suspend_path
        self.force_resume = force_resume

        self.__set_with_environment_variable(
            force_resume, "force_resume", "QUOP_FORCE_RESUME", int
        )

        self.__set_with_environment_variable(
            max_depths,
            "depths",
            "QUOP_MAX_DEPTH",
            lambda maxdepth: range(1, int(maxdepth) + 1),
        )
        self.__set_with_environment_variable(
            repeats,
            "repeats",
            "QUOP_REPEATS",
            lambda repeats: range(1, int(repeats) + 1),
        )

        self.__set_with_environment_variable(
            time_limit, "time_limit", "QUOP_TIME_LIMIT", float
        )
        self.__set_with_environment_variable(
            suspend_path, "suspend_path", "QUOP_SUSPEND_PATH", str
        )

        self.suspend_path = (
            ensure_path_and_extension(suspend_path, "quop")
            if self.suspend_path is not None
            else self.suspend_path
        )

        self.suspend_dict = {}
        self.source = self.__get_source()
        self.got_match = False
        self.complete = False
        self.marked_time = None
        self.job_index = -1  # allow for first increment to job_index = 0

        if self.suspend_path is not None:
            self.__get_match()

        self.__mark_time()
        if (self.time_limit is not None) and self.got_match:
            self.__resume()
        else:
            self.complete = False
            self.job_list = None
            self.results_dict = {depth: [] for depth in self.depths}

        self.__gen_job_list()
        self.__get_job_index()

    def get_job(self):
        self.__mark_time()
        return self.job_list[self.job_index]

    def get_seed(self):
        return self.seed

    def get_results(self):
        return self.results_dict

    def update(self, result):

        self.MPI_COMM.barrier()

        if self.MPI_COMM.Get_rank() == 0:
            depth_key = self.job_list[self.job_index][1]
            self.results_dict[depth_key].append(result)

        self.__get_job_index()
        self.__update_seed()
        self.__dump()

        if self.complete and (self.suspend_path is not None):
            root_print(self.MPI_COMM, "Job complete.")

        if (
            (self.time_limit is not None) and (not self.complete)
        ) and self.suspend_path is not None:

            self.__update_time()

            if not self.__enough_time():

                root_print(
                    self.MPI_COMM,
                    f"Time limit reached suspending to '{self.suspend_path}'.",
                )
                self.__suspend()

    def __set_with_environment_variable(
        self, source_value, attribute, environment_variable, typecast
    ):

        env = environ.get(environment_variable)

        if not env is None:

            env = typecast(env)

            if self.MPI_COMM.Get_rank() == 0:
                info(
                    (
                        f"Setting '{attribute}' to '{env}' from "
                        f"environment variable '{environment_variable}'"
                    )
                )

            setattr(self, attribute, env)


    def __mark_time(self):

        if self.MPI_COMM.Get_rank() == 0:
            self.marked_time = time()
        else:
            self.marked_time = None

        self.marked_time = self.MPI_COMM.bcast(self.marked_time, 0)

    def __update_seed(self):
        self.seed += 1

    def __update_time(self):

        self.time_limit -= time() - self.marked_time
        self.time_limit = self.MPI_COMM.allreduce(self.time_limit, op=MPI.MIN)

    def __enough_time(self):
        if not self.time_limit is None:
            return self.time_limit > 2 * (time() - self.marked_time)
        else:
            return True

    def __get_source(self):
        source = inspect.getsource(__main__)
        source = source.splitlines()
        source = [f"Line {i}: {line.strip()}" for i, line in enumerate(source)]
        return "\n".join([s for s in source if s.strip("\r\n").strip()])

    def __is_matching_source(self, source, get_lines=False):

        for current_line, old_line in zip(
            self.source.splitlines(), source.splitlines()
        ):

            if not current_line == old_line:
                if get_lines:
                    return False, current_line, old_line
                else:
                    return False, None, None

        return True, None, None

    def __load_suspened_file(self, path):
        with open(path, "rb") as f:
            suspend_dict = pickle.load(f)
        return suspend_dict

    def __get_match(self):

        got_match = False

        if self.MPI_COMM.Get_rank() == 0:

            suspend_path = self.suspend_path
            suspend_dict = self.suspend_dict

            exists = path.exists(suspend_path)
            if exists:
                suspend_dict = self.__load_suspened_file(suspend_path)
                got_match, lineold, linenew = self.__is_matching_source(
                    suspend_dict["source"], get_lines=True
                )
                if got_match or (self.force_resume == 1):
                    got_match = True
                    root_print(self.MPI_COMM, f"Resuming from '{suspend_path}'.")

            if not got_match:
                suspend_files = glob(f"{path.dirname(suspend_path)}*.quop")

                for sfile in suspend_files:
                    suspend_dict = self.__load_suspened_file(sfile)
                    suspend_path = sfile
                    if self.__is_matching_source(suspend_dict["source"])[0]:
                        got_match = True
                        info(f"Resuming from {sfile}.")
                        break

            if (not got_match) and exists:
                suspend_dict = self.__load_suspened_file(self.suspend_path)
                lines = self.__is_matching_source(
                    suspend_dict["source"], get_lines=True
                )[1:]
                suspend_path = ensure_path_and_extension(
                    self.suspend_path, "quop", add_time=True, unique=True
                )

                warn(
                    (
                        f"Detected changes to '{__main__.__file__}'. "
                        f"\n{lines[0]}\nDoes not match:\n{lines[1]}\n"
                        f"Restarting job and suspending "
                        f"to '{suspend_path}'. Force resume from "
                        f"'{self.suspend_path}' by defining the environment "
                        f"variable: 'QUOP_FORCE_RESUME=1'"
                    )
                )
        else:
            suspend_path = None
            suspend_dict = None

        self.got_match = self.MPI_COMM.bcast(got_match, root=0)
        self.suspend_path = self.MPI_COMM.bcast(suspend_path, root=0)
        self.suspend_dict = self.MPI_COMM.bcast(suspend_dict, root=0)

    def __gen_job_list(self):
        self.job_list = [
            [repeat, depth] for depth in self.depths for repeat in self.repeats
        ]

    def __get_job_index(self):

        maxrepeats = self.job_list[-1][0] + 1
        self.job_index += 1

        if self.job_index == len(self.job_list):
            self.complete = True
            return

        else:
            if (
                len(self.results_dict[self.job_list[self.job_index][1]])
                > self.job_list[self.job_index][0] - 1
            ):
                while (
                    len(self.results_dict[self.job_list[self.job_index][1]])
                    > self.job_list[self.job_index][0] - 1
                ):
                    self.job_index += 1
                    if self.job_index == len(self.job_list):
                        self.complete = True
                        return

    def __dump(self):

        if self.MPI_COMM.Get_rank() == 0 and not self.time_limit is None:
            d = {"source": self.__get_source()}
            for member in inspect.getmembers(self):

                ignore = [
                    "__",
                    "MPI_COMM",
                    "got_match",
                    "time_limit",
                    "marked_time",
                    "job_list",
                    "job_index",
                    "suspend_path",
                    "repeats",
                    "depths",
                ]
                if all([not (ig in member[0]) for ig in ignore]) and (
                    not inspect.ismethod(member[1])
                ):
                    d[member[0]] = member[1]

            with open(self.suspend_path, "wb") as f:
                pickle.dump(d, f)

    def __suspend(self):

        self.__dump()
        self.MPI_COMM.barrier()
        exit(0)

    def __resume(self):

        for key in self.suspend_dict.keys():
            setattr(self, key, self.suspend_dict[key])

        if self.complete:
            root_print(self.MPI_COMM, "Job complete.")

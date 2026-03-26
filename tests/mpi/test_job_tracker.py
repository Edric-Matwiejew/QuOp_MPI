"""
Tests for JobTracker class used by benchmark() method.

The JobTracker manages:
- Job list generation (repeat, depth pairs)
- Progress tracking through jobs
- Suspend/resume functionality via pickle files
- Time limit handling
- Seed management for reproducibility

Run with: mpiexec -n 2 python -m pytest tests/mpi/test_job_tracker.py -v --with-mpi
"""

import os
import pickle
import tempfile

import pytest

@pytest.mark.mpi
class TestJobTrackerBasicInit:
    """Tests for basic JobTracker initialization."""

    def test_init_creates_job_list(self, mpi_comm):
        """Verify JobTracker creates a job list on init."""
        from quop_mpi._utils._tracker import JobTracker

        repeats = 3
        max_depth = 2

        tracker = JobTracker(
            repeats=repeats,
            max_depths=max_depth,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # Job list should be created
        assert tracker.job_list is not None
        assert len(tracker.job_list) == repeats * max_depth

    def test_job_list_structure(self, mpi_comm):
        """Verify job list contains [repeat, depth] pairs."""
        from quop_mpi._utils._tracker import JobTracker

        repeats = 2
        max_depth = 3

        tracker = JobTracker(
            repeats=repeats,
            max_depths=max_depth,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # Jobs should be ordered by depth, then by repeat
        expected_jobs = [
            [1, 1],
            [2, 1],  # depth 1
            [1, 2],
            [2, 2],  # depth 2
            [1, 3],
            [2, 3],  # depth 3
        ]

        assert tracker.job_list == expected_jobs

    def test_init_not_complete(self, mpi_comm):
        """Verify tracker starts as not complete."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        assert not tracker.complete

    def test_init_creates_results_dict(self, mpi_comm):
        """Verify results_dict is initialized for all depths."""
        from quop_mpi._utils._tracker import JobTracker

        max_depth = 4
        tracker = JobTracker(
            repeats=2,
            max_depths=max_depth,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # results_dict should have keys for each depth
        for depth in range(1, max_depth + 1):
            assert depth in tracker.results_dict
            assert tracker.results_dict[depth] == []

    def test_init_with_seed(self, mpi_comm):
        """Verify seed is properly stored."""
        from quop_mpi._utils._tracker import JobTracker

        seed = 42
        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=seed,
            suspend_path=None,
        )

        assert tracker.seed == seed


@pytest.mark.mpi
class TestJobTrackerGetJob:
    """Tests for get_job() method."""

    def test_get_job_returns_first_job(self, mpi_comm):
        """Verify get_job returns the first job."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        job = tracker.get_job()

        # First job should be [repeat=1, depth=1]
        assert job == [1, 1]

    def test_get_job_consistent_across_ranks(self, mpi_comm):
        """Verify all ranks get the same job."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        local_job = tracker.get_job()

        # Gather jobs from all ranks
        all_jobs = mpi_comm.gather(local_job, root=0)

        if mpi_comm.Get_rank() == 0:
            # All ranks should have the same job
            for job in all_jobs:
                assert job == all_jobs[0]


@pytest.mark.mpi
class TestJobTrackerUpdate:
    """Tests for update() method."""

    def test_update_advances_job_index(self, mpi_comm):
        """Verify update() moves to next job."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # Get first job
        first_job = tracker.get_job()
        assert first_job == [1, 1]

        # Update with a dummy result
        mock_result = {"fun": 1.0, "x": [0.1, 0.2]}
        tracker.update(mock_result)

        # Get second job
        second_job = tracker.get_job()
        assert second_job == [2, 1]  # Next repeat at depth 1

    def test_update_stores_result(self, mpi_comm):
        """Verify update() stores the result in results_dict."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # Get and complete first job
        tracker.get_job()
        mock_result = {"fun": 1.5, "x": [0.3, 0.4]}
        tracker.update(mock_result)

        if mpi_comm.Get_rank() == 0:
            # Result should be stored at depth 1
            assert len(tracker.results_dict[1]) == 1
            assert tracker.results_dict[1][0] == mock_result

    def test_update_increments_seed(self, mpi_comm):
        """Verify update() increments the seed."""
        from quop_mpi._utils._tracker import JobTracker

        initial_seed = 10
        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=initial_seed,
            suspend_path=None,
        )

        tracker.get_job()
        tracker.update({"fun": 1.0})

        assert tracker.seed == initial_seed + 1

    def test_complete_after_all_jobs(self, mpi_comm):
        """Verify tracker is complete after all jobs processed."""
        from quop_mpi._utils._tracker import JobTracker

        repeats = 2
        max_depth = 2

        tracker = JobTracker(
            repeats=repeats,
            max_depths=max_depth,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # Process all jobs
        for _ in range(repeats * max_depth):
            assert not tracker.complete
            tracker.get_job()
            tracker.update({"fun": 1.0})

        assert tracker.complete


@pytest.mark.mpi
class TestJobTrackerGetSeed:
    """Tests for get_seed() method."""

    def test_get_seed_returns_current_seed(self, mpi_comm):
        """Verify get_seed returns the current seed value."""
        from quop_mpi._utils._tracker import JobTracker

        seed = 123
        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=seed,
            suspend_path=None,
        )

        assert tracker.get_seed() == seed

    def test_get_seed_after_updates(self, mpi_comm):
        """Verify get_seed reflects updates."""
        from quop_mpi._utils._tracker import JobTracker

        seed = 0
        tracker = JobTracker(
            repeats=3,
            max_depths=1,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=seed,
            suspend_path=None,
        )

        for i in range(3):
            assert tracker.get_seed() == seed + i
            tracker.get_job()
            tracker.update({"fun": 1.0})


@pytest.mark.mpi
class TestJobTrackerGetResults:
    """Tests for get_results() method."""

    def test_get_results_returns_dict(self, mpi_comm):
        """Verify get_results returns the results dictionary."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        results = tracker.get_results()

        assert isinstance(results, dict)
        assert 1 in results
        assert 2 in results

    def test_get_results_after_jobs(self, mpi_comm):
        """Verify get_results contains completed job results."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=2,
            max_depths=1,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # Complete two jobs
        tracker.get_job()
        tracker.update({"fun": 1.0, "id": "job1"})
        tracker.get_job()
        tracker.update({"fun": 2.0, "id": "job2"})

        results = tracker.get_results()

        if mpi_comm.Get_rank() == 0:
            assert len(results[1]) == 2
            assert results[1][0]["id"] == "job1"
            assert results[1][1]["id"] == "job2"


@pytest.mark.mpi
class TestJobTrackerSuspendResume:
    """Tests for suspend/resume functionality."""

    def test_suspend_path_creates_file(self, mpi_comm):
        """Verify suspend creates a file when time_limit is set."""
        from quop_mpi._utils._tracker import JobTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            suspend_path = os.path.join(tmpdir, "test_suspend")

            tracker = JobTracker(
                repeats=2,
                max_depths=2,
                time_limit=3600,  # 1 hour - won't trigger suspend
                MPI_COMM=mpi_comm,
                seed=0,
                suspend_path=suspend_path,
            )

            # Complete a job to trigger dump
            tracker.get_job()
            tracker.update({"fun": 1.0})

            # File should be created (with .quop extension)
            expected_file = suspend_path + ".quop"
            if mpi_comm.Get_rank() == 0:
                assert os.path.exists(expected_file)

            mpi_comm.barrier()

    def test_suspend_file_contains_state(self, mpi_comm):
        """Verify suspend file contains tracker state."""
        from quop_mpi._utils._tracker import JobTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            suspend_path = os.path.join(tmpdir, "test_suspend")

            tracker = JobTracker(
                repeats=2,
                max_depths=2,
                time_limit=3600,
                MPI_COMM=mpi_comm,
                seed=42,
                suspend_path=suspend_path,
            )

            tracker.get_job()
            tracker.update({"fun": 1.5})

            mpi_comm.barrier()

            if mpi_comm.Get_rank() == 0:
                expected_file = suspend_path + ".quop"
                with open(expected_file, "rb") as f:
                    data = pickle.load(f)

                # Should contain key state
                assert "seed" in data
                assert "results_dict" in data
                assert "source" in data

    def test_no_suspend_without_time_limit(self, mpi_comm):
        """Verify no suspend file is created without time_limit."""
        from quop_mpi._utils._tracker import JobTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            suspend_path = os.path.join(tmpdir, "test_suspend")

            tracker = JobTracker(
                repeats=2,
                max_depths=2,
                time_limit=None,  # No time limit
                MPI_COMM=mpi_comm,
                seed=0,
                suspend_path=suspend_path,
            )

            tracker.get_job()
            tracker.update({"fun": 1.0})

            mpi_comm.barrier()

            # File should NOT be created
            expected_file = suspend_path + ".quop"
            if mpi_comm.Get_rank() == 0:
                assert not os.path.exists(expected_file)


@pytest.mark.mpi
class TestJobTrackerJobProgression:
    """Tests for correct job progression through depths and repeats."""

    def test_progression_order(self, mpi_comm):
        """Verify jobs progress in correct order: all repeats at depth 1, then depth 2, etc."""
        from quop_mpi._utils._tracker import JobTracker

        repeats = 3
        max_depth = 2

        tracker = JobTracker(
            repeats=repeats,
            max_depths=max_depth,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        jobs_seen = []
        while not tracker.complete:
            job = tracker.get_job()
            jobs_seen.append(job.copy())
            tracker.update({"fun": 1.0})

        # Expected order
        expected = [
            [1, 1],
            [2, 1],
            [3, 1],  # depth 1
            [1, 2],
            [2, 2],
            [3, 2],  # depth 2
        ]

        assert jobs_seen == expected

    def test_single_repeat_single_depth(self, mpi_comm):
        """Test minimal case: 1 repeat, 1 depth."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=1,
            max_depths=1,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        assert not tracker.complete

        job = tracker.get_job()
        assert job == [1, 1]

        tracker.update({"fun": 1.0})
        assert tracker.complete

    def test_many_depths_few_repeats(self, mpi_comm):
        """Test with many depths but few repeats."""
        from quop_mpi._utils._tracker import JobTracker

        repeats = 1
        max_depth = 5

        tracker = JobTracker(
            repeats=repeats,
            max_depths=max_depth,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        depths_seen = []
        while not tracker.complete:
            job = tracker.get_job()
            depths_seen.append(job[1])
            tracker.update({"fun": 1.0})

        assert depths_seen == [1, 2, 3, 4, 5]


@pytest.mark.mpi
class TestJobTrackerEdgeCases:
    """Tests for edge cases."""

    def test_large_repeats(self, mpi_comm):
        """Test with many repeats."""
        from quop_mpi._utils._tracker import JobTracker

        repeats = 10
        max_depth = 1

        tracker = JobTracker(
            repeats=repeats,
            max_depths=max_depth,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        count = 0
        while not tracker.complete:
            tracker.get_job()
            tracker.update({"fun": 1.0})
            count += 1

        assert count == repeats

    def test_results_accessible_during_run(self, mpi_comm):
        """Verify results can be accessed while tracker is running."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=3,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # Complete first job
        tracker.get_job()
        tracker.update({"fun": 0.5})

        # Access results mid-run
        results = tracker.get_results()

        if mpi_comm.Get_rank() == 0:
            assert len(results[1]) == 1
            assert results[1][0]["fun"] == 0.5

    def test_seed_unique_per_job(self, mpi_comm):
        """Verify each job gets a unique seed."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=3,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=100,
            suspend_path=None,
        )

        seeds_seen = []
        while not tracker.complete:
            seeds_seen.append(tracker.get_seed())
            tracker.get_job()
            tracker.update({"fun": 1.0})

        # All seeds should be unique
        assert len(seeds_seen) == len(set(seeds_seen))
        # Seeds should be consecutive
        assert seeds_seen == list(range(100, 100 + len(seeds_seen)))


@pytest.mark.mpi
class TestJobTrackerMPIConsistency:
    """Tests for MPI consistency across ranks."""

    def test_job_index_consistent(self, mpi_comm):
        """Verify job_index is consistent across ranks."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=2,
            max_depths=2,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        # Complete a few jobs
        for _ in range(3):
            tracker.get_job()
            tracker.update({"fun": 1.0})

        # All ranks should have same job_index
        local_index = tracker.job_index
        all_indices = mpi_comm.gather(local_index, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(idx == all_indices[0] for idx in all_indices)

    def test_complete_flag_consistent(self, mpi_comm):
        """Verify complete flag is consistent across ranks."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=1,
            max_depths=1,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        tracker.get_job()
        tracker.update({"fun": 1.0})

        # All ranks should agree on completion
        local_complete = tracker.complete
        all_complete = mpi_comm.gather(local_complete, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(c for c in all_complete)

    def test_seed_consistent_across_ranks(self, mpi_comm):
        """Verify seed is consistent across ranks after updates."""
        from quop_mpi._utils._tracker import JobTracker

        tracker = JobTracker(
            repeats=3,
            max_depths=1,
            time_limit=None,
            MPI_COMM=mpi_comm,
            seed=0,
            suspend_path=None,
        )

        for _ in range(2):
            tracker.get_job()
            tracker.update({"fun": 1.0})

        local_seed = tracker.seed
        all_seeds = mpi_comm.gather(local_seed, root=0)

        if mpi_comm.Get_rank() == 0:
            assert all(s == all_seeds[0] for s in all_seeds)

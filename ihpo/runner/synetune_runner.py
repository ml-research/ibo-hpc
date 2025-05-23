from syne_tune.backend.trial_backend import TrialBackend
from syne_tune.backend.trial_status import Status
from pathlib import Path
import time

class SyneTuneRunner(TrialBackend):

    def __init__(self, objective):
        self.objective = objective
        self._sync_trial_results = {}
        self.delete_checkpoints = False
        super().__init__()

    def copy_checkpoint(self, src_trial_id: int, tgt_trial_id: int):
        """
        Copy the checkpoint folder from one trial to the other.

        :param src_trial_id: Source trial ID (copy from)
        :param tgt_trial_id: Target trial ID (copy to)
        """
        pass

    def delete_checkpoint(self, trial_id: int):
        """
        Removes checkpoint folder for a trial. It is OK for the folder not to
        exist.

        :param trial_id: ID of trial for which checkpoint files are deleted
        """
        pass

    def _resume_trial(self, trial_id: int):
        """Called in :meth:`resume_trial`, before job is scheduled.

        :param trial_id: See ``resume_trial``
        """
        pass

    def _pause_trial(self, trial_id: int, result):
        """Implements :meth:`pause_trial`.

        :param trial_id: ID of trial to pause
        :param result: Result dict based on which scheduler decided to pause the
            trial
        """
        pass

    def _cleanup_after_trial(self, trial_id: int):
        """
        This is called whenever a trial is stopped or paused.
        Note that ``delete_checkpoints`` should not be dealt with here, since
        checkpoints must not be deleted when a trial is paused.

        :param trial_id: ID of trial to clean up after
        """
        pass

    def _stop_trial(self, trial_id: int, result):
        """Backend specific operation that stops the trial.

        :param trial_id: ID of trial to stop
        :param result: Result dict based on which scheduler decided to stop the
            trial
        """
        pass

    def _schedule(self, trial_id: int, config):
        """Schedules job for trial evaluation.

        Called by :meth:`start_trial`, :meth:`resume_trial`.

        :param trial_id: ID of trial to schedule
        :param config: Configuration for this trial
        """
        res = self.objective(config)
        self._sync_trial_results[trial_id] = res.val_performance

    def _all_trial_results(self, trial_ids):
        """Returns results for selected trials

        :param trial_ids: IDs of trials for which results are to be queried
        :return: list of results corresponding to ``trial_ids``, contains all the
            results obtained since the start of each trial.
        """
        trials = []
        for trial_id in trial_ids:
            score = self._sync_trial_results[trial_id]
            trial_obj = self._trial_dict[trial_id]
            trial_obj.status = Status.completed
            trial_obj.metrics = [{'st_worker_timestamp': time.time(),'val_performance': score}]
            trials.append(trial_obj)

        return trials


    def busy_trial_ids(self):
        """Returns list of ids for currently busy trials

        A trial is busy if its status is
        :const:`~syne_tune.backend.trial_status.Status.in_progress` or
        :const:`~syne_tune.backend.trial_status.Status.stopping`.
        If the execution setup is able to run ``n_workers`` jobs in parallel,
        then if this method returns a list of size ``n``, the tuner may start
        ``n_workers - n`` new jobs.

        :return: List of ``(trial_id, status)``
        """
        return []

    def stdout(self, trial_id: int):
        """Fetch ``stdout`` log for trial

        :param trial_id: ID of trial
        :return: Lines of the log of the trial (stdout)
        """
        return f"Running {trial_id}"

    def stderr(self, trial_id: int):
        """Fetch ``stderr`` log for trial

        :param trial_id: ID of trial
        :return: Lines of the log of the trial (stderr)
        """
        return f"Trial {trial_id} failed"

    def set_path(
        self, results_root = None, tuner_name = None
    ):
        """
        :param results_root: The local folder that should contain the results of
            the tuning experiment. Used by :class:`~syne_tune.Tuner` to indicate
            a desired path where the results should be written to. This is used
            to unify the location of backend files and :class:`~syne_tune.Tuner`
            results when possible (in the local backend). By default, the backend
            does not do anything since not all backends may be able to unify their
            file locations.
        :param tuner_name: Name of the tuner, can be used for instance to save
            checkpoints on remote storage.
        """
        pass

    def entrypoint_path(self):
        """
        :return: Entrypoint path of script to be executed
        """
        return Path('dummy')

    def set_entrypoint(self, entry_point: str):
        """Update the entrypoint.

        :param entry_point: New path of the entrypoint.
        """
        pass

    def on_tuner_save(self):
        """
        Called at the end of :meth:`~syne_tune.Tuner.save`.
        """
        pass
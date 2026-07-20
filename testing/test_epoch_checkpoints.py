import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from api_server.epoch_controller import StepBudgetController
from api_server.app import SessionCreateRequest
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


class StepBudgetControllerTests(unittest.TestCase):
    def test_pause_is_published_after_checkpoint_is_saved(self):
        notifications = []
        controller = StepBudgetController(
            total_steps=4,
            on_budget_exhausted=lambda step, epoch: notifications.append((step, epoch)),
        )

        self.assertEqual(controller.allow_steps(2), 2)
        self.assertEqual(controller.on_step_end(1, 1), "continue")
        self.assertEqual(controller.on_step_end(2, 1), "pause")
        self.assertFalse(controller.pause_event.is_set())
        self.assertEqual(notifications, [])

        controller.notify_checkpoint_saved(2, 1)

        self.assertTrue(controller.pause_event.is_set())
        self.assertEqual(notifications, [(2, 1)])

    def test_final_budget_completes_after_checkpoint_is_saved(self):
        controller = StepBudgetController(total_steps=2)

        self.assertEqual(controller.allow_steps(2), 2)
        self.assertEqual(controller.on_step_end(2, 1), "pause")
        self.assertFalse(controller.pause_event.is_set())

        controller.notify_checkpoint_saved(2, 1)

        self.assertEqual(controller.wait_for_resume(), "complete")

    def test_checkpoint_step_must_match_completed_step(self):
        controller = StepBudgetController(total_steps=2)
        controller.allow_steps(2)
        controller.on_step_end(2, 1)

        with self.assertRaisesRegex(ValueError, "checkpoint step 1 does not match completed step 2"):
            controller.notify_checkpoint_saved(1, 1)


class SessionCreateRequestTests(unittest.TestCase):
    def test_snake_case_session_fields_are_accepted(self):
        request = SessionCreateRequest.model_validate({
            "session_id": "session-id",
            "max_steps": 4,
            "config": {},
        })

        self.assertEqual(request.session_id, "session-id")
        self.assertEqual(request.max_steps, 4)


class _BlockingNetwork:
    def __init__(self):
        self.multiplier = 1.0
        self.save_started = threading.Event()
        self.release_save = threading.Event()

    def save_weights(self, path, **_kwargs):
        self.save_started.set()
        self.release_save.wait(timeout=5)
        with open(path, "wb") as checkpoint_file:
            checkpoint_file.write(b"checkpoint")


class CheckpointSaveTests(unittest.TestCase):
    def test_last_save_step_updates_after_checkpoint_write(self):
        with tempfile.TemporaryDirectory() as save_root:
            trainer = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
            trainer.accelerator = SimpleNamespace(is_main_process=True)
            trainer.ema = None
            trainer.save_root = save_root
            trainer.job = SimpleNamespace(name="test")
            trainer.last_save_step = 3
            trainer.meta = {}
            trainer.adapter = None
            trainer.is_fine_tuning = False
            trainer.train_config = SimpleNamespace(merge_network_on_save=False)
            trainer.network = _BlockingNetwork()
            trainer.named_lora = False
            trainer.embedding = None
            trainer.save_config = SimpleNamespace(dtype="float16")
            trainer.decorator = None
            trainer.adapter_config = None
            trainer.snr_gos = None
            trainer.optimizer = None
            trainer.update_training_metadata = lambda: None
            trainer.clean_up_saves = lambda: None
            trainer.post_save_hook = lambda _path: None

            with (
                patch("jobs.process.BaseSDTrainProcess.flush"),
                patch("jobs.process.BaseSDTrainProcess.get_meta_for_safetensors", return_value={}),
            ):
                save_thread = threading.Thread(target=trainer.save, args=(7,))
                save_thread.start()
                self.assertTrue(trainer.network.save_started.wait(timeout=5))
                self.assertEqual(trainer.last_save_step, 3)

                trainer.network.release_save.set()
                save_thread.join(timeout=5)

            self.assertFalse(save_thread.is_alive())
            self.assertEqual(trainer.last_save_step, 7)


if __name__ == "__main__":
    unittest.main()

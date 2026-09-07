import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from api_server.tagger import (
    CL_TAGGER_DEFAULT_THRESHOLD,
    WD14_DEFAULT_CHARACTER_THRESHOLD,
    WD14_DEFAULT_GENERAL_THRESHOLD,
    MediaType,
    PredictionRequest,
    PredictionResult,
    TaggerService,
    load_labels,
    resolve_model_and_labels,
    tag_image,
)
from api_server.tagging_models import (
    CL_TAGGER_DATA_FILENAME,
    CL_TAGGER_MODEL_ID,
    WD14_MODEL_ID,
    get_model_and_labels,
)


class _Node:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self, output, input_shape, providers=None):
        self.output = np.asarray([output], dtype=np.float32)
        self.input_shape = input_shape
        self.received_input = None
        self.providers = list(providers or ["CPUExecutionProvider"])
        self.provider_changes = []
        self.run_count = 0

    def get_providers(self):
        return list(self.providers)

    def set_providers(self, providers):
        self.providers = list(providers)
        self.provider_changes.append(list(providers))

    def get_inputs(self):
        return [_Node("pixel_values", self.input_shape)]

    def get_outputs(self):
        return [_Node("scores")]

    def run(self, output_names, inputs):
        self.run_count += 1
        self.received_input = inputs["pixel_values"]
        return [self.output]


class TaggerTests(unittest.TestCase):
    def test_registry_supports_wd14_and_cl_tagger_identifiers(self):
        def fake_download(repo, filename, revision):
            return f"/cache/{repo}/{filename}@{revision}"

        with patch(
            "api_server.tagging_models.huggingface_hub.hf_hub_download",
            side_effect=fake_download,
        ) as download:
            wd_model, wd_labels = get_model_and_labels(WD14_MODEL_ID)
            cl_model, cl_labels = get_model_and_labels(CL_TAGGER_MODEL_ID)

        self.assertIn("SmilingWolf/wd-v1-4-vit-tagger/model.onnx", wd_model)
        self.assertIn("selected_tags.csv", wd_labels)
        self.assertIn("cella110n/cl_tagger_v2/v2_00/model.onnx", cl_model)
        self.assertIn("model_vocabulary.json", cl_labels)
        cl_filenames = [call.args[1] for call in download.call_args_list[2:]]
        self.assertEqual(
            [
                f"v2_00/{CL_TAGGER_DATA_FILENAME}",
                "v2_00/model.onnx",
                "v2_00/model_vocabulary.json",
            ],
            cl_filenames,
        )

    def test_prediction_request_defers_to_model_specific_thresholds(self):
        request = PredictionRequest.new("image.png", MediaType.image)
        explicit = PredictionRequest.from_dict(
            {
                "media_path": "image.png",
                "general_threshold": 0.2,
                "character_threshold": 0.7,
            }
        )

        self.assertIsNone(request.general_threshold)
        self.assertIsNone(request.character_threshold)
        self.assertEqual(0.2, explicit.general_threshold)
        self.assertEqual(0.7, explicit.character_threshold)

    def test_load_labels_reads_cl_tagger_vocabulary_categories(self):
        vocabulary = {
            "num_tags": 5,
            "idx_to_tag": {
                "0": "general",
                "1": "explicit",
                "2": "1girl",
                "3": "hatsune miku",
                "4": "ignored copyright",
            },
            "tag_to_category": {
                "general": "Rating",
                "explicit": "Rating",
                "1girl": "General",
                "hatsune miku": "Character",
                "ignored copyright": "Copyright",
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            vocabulary_path = Path(directory) / "model_vocabulary.json"
            vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
            labels = load_labels(str(vocabulary_path))

        self.assertEqual(list(vocabulary["idx_to_tag"].values()), labels[0])
        self.assertEqual([0, 1], labels[1])
        self.assertEqual([2], labels[2])
        self.assertEqual([3], labels[3])

    def test_load_labels_reads_wd14_csv_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            labels_path = Path(directory) / "selected_tags.csv"
            labels_path.write_text(
                "tag_id,name,category,count\n"
                "0,general,9,1\n"
                "1,1girl,0,1\n"
                "2,hatsune_miku,4,1\n",
                encoding="utf-8",
            )
            labels = load_labels(str(labels_path))

        self.assertEqual(["general", "1girl", "hatsune_miku"], labels[0])
        self.assertEqual([0], labels[1])
        self.assertEqual([1], labels[2])
        self.assertEqual([2], labels[3])

    def test_resolve_cl_tagger_supports_packed_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory) / "v2_00"
            model_directory.mkdir()
            (model_directory / "model.onnx").touch()
            (model_directory / "model.onnx.data").touch()
            (model_directory / "model_vocabulary.json").touch()

            model_path, vocabulary_path = resolve_model_and_labels(
                CL_TAGGER_MODEL_ID, directory
            )

        self.assertEqual(str(model_directory / "model.onnx"), model_path)
        self.assertEqual(
            str(model_directory / "model_vocabulary.json"), vocabulary_path
        )

    def test_resolve_wd14_supports_packed_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "model.onnx").touch()
            (model_directory / "selected_tags.csv").touch()

            model_path, labels_path = resolve_model_and_labels(
                WD14_MODEL_ID, directory
            )

        self.assertEqual(str(model_directory / "model.onnx"), model_path)
        self.assertEqual(str(model_directory / "selected_tags.csv"), labels_path)

    def test_cl_tagger_uses_siglip_preprocessing_and_defaults(self):
        logits = [
            math.log(0.8 / 0.2),
            math.log(0.2 / 0.8),
            math.log(0.9 / 0.1),
            0.0,
            math.log(0.6 / 0.4),
        ]
        model = _FakeSession(logits, ["batch_size", 3, 2, 2])
        labels = (
            ["general", "explicit", "1girl", "low_score", "hatsune_miku"],
            [0, 1],
            [2, 3],
            [4],
        )
        image = Image.new("RGB", (4, 2), (255, 0, 0))

        result = tag_image(image, model, labels)

        self.assertEqual(0.55, CL_TAGGER_DEFAULT_THRESHOLD)
        self.assertEqual((1, 3, 2, 2), model.received_input.shape)
        np.testing.assert_allclose(model.received_input[:, 0], 1.0)
        np.testing.assert_allclose(model.received_input[:, 1:], -1.0)
        self.assertAlmostEqual(0.8, result["rating"]["general"], places=6)
        self.assertAlmostEqual(0.2, result["rating"]["explicit"], places=6)
        self.assertEqual(["1girl"], list(result["tags"]))
        self.assertEqual(["hatsune miku"], list(result["characters"]))

    def test_wd14_uses_nhwc_bgr_preprocessing_and_original_defaults(self):
        model = _FakeSession(
            [0.8, 0.2, 0.4, 0.3, 0.9],
            ["batch_size", 2, 2, 3],
        )
        labels = (
            ["general", "explicit", "1girl", "low_score", "hatsune_miku"],
            [0, 1],
            [2, 3],
            [4],
        )
        image = Image.new("RGB", (2, 2), (255, 0, 0))

        result = tag_image(image, model, labels)

        self.assertEqual(0.35, WD14_DEFAULT_GENERAL_THRESHOLD)
        self.assertEqual(0.85, WD14_DEFAULT_CHARACTER_THRESHOLD)
        self.assertEqual((1, 2, 2, 3), model.received_input.shape)
        np.testing.assert_allclose(model.received_input[..., 0:2], 0.0)
        np.testing.assert_allclose(model.received_input[..., 2], 255.0)
        self.assertEqual(["1girl"], list(result["tags"]))
        self.assertEqual(["hatsune miku"], list(result["characters"]))

    def test_explicit_thresholds_override_each_models_defaults(self):
        cl_model = _FakeSession([0.0], ["batch_size", 3, 2, 2])
        wd_model = _FakeSession([0.3], ["batch_size", 2, 2, 3])
        labels = (["tag"], [], [0], [])
        image = Image.new("RGB", (2, 2), (0, 0, 0))

        cl_result = tag_image(image, cl_model, labels, general_threshold=0.5)
        wd_result = tag_image(image, wd_model, labels, general_threshold=0.2)

        self.assertEqual({"tag": 0.5}, cl_result["tags"])
        self.assertAlmostEqual(0.3, wd_result["tags"]["tag"], places=6)

    def test_predict_inputs_includes_characters(self):
        service = TaggerService.__new__(TaggerService)
        service.predict_request = lambda request: [
            PredictionResult(
                request.media_path,
                True,
                tags={"1girl": 0.9},
                characters={"hatsune miku": 0.8},
                rating={"general": 0.7},
            )
        ]
        request = PredictionRequest.new("image.png")

        result = service.predict_inputs([request])["image.png"]

        self.assertEqual({"1girl": 0.9}, result["tags"])
        self.assertEqual({"hatsune miku": 0.8}, result["characters"])
        self.assertEqual({"general": 0.7}, result["rating"])


class TaggerSessionReuseTests(unittest.TestCase):
    CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    CPU_PROVIDERS = ["CPUExecutionProvider"]

    def setUp(self):
        self.labels = (["general", "tag", "character"], [0], [1], [2])
        self.requests = [
            PredictionRequest.new("first.png"),
            PredictionRequest.new("second.png"),
        ]
        images = patch(
            "api_server.tagger.images_from_data",
            return_value=[Image.new("RGB", (2, 2), "white")],
        )
        images.start()
        self.addCleanup(images.stop)

    def make_service(self, device, input_shape):
        providers = self.CUDA_PROVIDERS if device == "cuda" else self.CPU_PROVIDERS
        session = _FakeSession([0.9, 0.9, 0.9], input_shape, providers)
        with patch("api_server.tagger.load", return_value=(session, self.labels)):
            return TaggerService(device=torch.device(device))

    def test_repeated_and_multi_image_requests_reuse_session_for_both_models(self):
        for device in ["cpu", "cuda"]:
            for model, shape in [("wd14", [1, 2, 2, 3]), ("cl", [1, 3, 2, 2])]:
                with self.subTest(device=device, model=model):
                    service = self.make_service(device, shape)
                    expected = service.predict_inputs(self.requests)
                    self.assertTrue(expected["first.png"]["tags"])

                    for _ in range(3):
                        self.assertEqual(expected, service.predict_inputs(self.requests))

                    self.assertEqual(8, service.session.run_count)
                    self.assertEqual([], service.session.provider_changes)

    def test_offload_is_idempotent_and_next_request_restores_cuda_only_once(self):
        service = self.make_service("cuda", [1, 3, 2, 2])
        expected = service.predict_inputs(self.requests)

        service.offload()
        service.offload()

        self.assertEqual(self.CPU_PROVIDERS, service.session.get_providers())
        self.assertEqual([self.CPU_PROVIDERS], service.session.provider_changes)
        self.assertEqual(torch.device("cuda"), service.device)
        for _ in range(3):
            self.assertEqual(expected, service.predict_inputs(self.requests))
        self.assertEqual(self.CUDA_PROVIDERS, service.session.get_providers())
        self.assertEqual(
            [self.CPU_PROVIDERS, self.CUDA_PROVIDERS],
            service.session.provider_changes,
        )

    def test_cpu_only_offload_and_predictions_do_not_recreate_session(self):
        service = self.make_service("cpu", [1, 2, 2, 3])
        expected = service.predict_inputs(self.requests)

        service.offload()
        service.offload()

        self.assertEqual(expected, service.predict_inputs(self.requests))
        self.assertEqual(torch.device("cpu"), service.device)
        self.assertEqual(self.CPU_PROVIDERS, service.session.get_providers())
        self.assertEqual([], service.session.provider_changes)

    def test_changed_inference_device_reconfigures_session_only_once(self):
        service = self.make_service("cuda", [1, 3, 2, 2])
        service.device = torch.device("cpu")

        service.predict_inputs(self.requests)
        service.predict_inputs(self.requests)

        self.assertEqual(self.CPU_PROVIDERS, service.session.get_providers())
        self.assertEqual([self.CPU_PROVIDERS], service.session.provider_changes)

    def test_api_model_switch_offloads_previous_model_and_reuses_current_model(self):
        from api_server import app as api

        sessions = [
            _FakeSession([0.9, 0.9, 0.9], shape, self.CUDA_PROVIDERS)
            for shape in [[1, 2, 2, 3], [1, 3, 2, 2], [1, 2, 2, 3]]
        ]
        with (
            patch.object(api, "_tagger_service", None),
            patch.object(api, "_tagger_model_path", None),
            patch(
                "api_server.tagger.resolve_device", return_value=torch.device("cuda")
            ),
            patch(
                "api_server.tagger.load",
                side_effect=[(session, self.labels) for session in sessions],
            ) as load_model,
        ):
            previous = None
            for index, path in enumerate(["/models/wd14", "/models/cl", "/models/wd14"]):
                service = api._get_tagger_service(path)
                self.assertIsNot(previous, service)
                self.assertIs(sessions[index], service.session)
                self.assertEqual(index + 1, load_model.call_count)
                self.assertEqual(path, load_model.call_args.kwargs["model_path"])

                self.assertIs(service, api._get_tagger_service(path))
                expected = service.predict_inputs(self.requests)
                self.assertEqual(expected, service.predict_inputs(self.requests))
                self.assertEqual([], service.session.provider_changes)
                if previous is not None:
                    self.assertEqual(
                        [self.CPU_PROVIDERS], previous.session.provider_changes
                    )
                previous = service


if __name__ == "__main__":
    unittest.main()

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from api_server.tagger import (
    DEFAULT_TAG_THRESHOLD,
    MediaType,
    PredictionRequest,
    PredictionResult,
    TaggerService,
    load_labels,
    resolve_model_and_labels,
    tag_image,
)
from api_server.tagging_models import get_model_and_labels


class _Node:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class _FakeSession:
    def __init__(self, logits):
        self.logits = np.asarray([logits], dtype=np.float32)
        self.received_input = None

    def get_inputs(self):
        return [_Node("pixel_values", ["batch_size", 3, 2, 2])]

    def get_outputs(self):
        return [_Node("logits")]

    def run(self, output_names, inputs):
        self.received_input = inputs["pixel_values"]
        return [self.logits]


class TaggerTests(unittest.TestCase):
    def test_wd14_identifier_is_not_kept_as_a_fallback(self):
        with self.assertRaisesRegex(ValueError, "Supported model: cl-tagger.v2"):
            get_model_and_labels("wd14-vit.v1")

    def test_prediction_request_uses_cl_tagger_recommended_threshold(self):
        request = PredictionRequest.new("image.png", MediaType.image)

        self.assertEqual(DEFAULT_TAG_THRESHOLD, request.general_threshold)
        self.assertEqual(DEFAULT_TAG_THRESHOLD, request.character_threshold)

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

    def test_resolve_model_supports_packed_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory) / "v2_00"
            model_directory.mkdir()
            (model_directory / "model.onnx").touch()
            (model_directory / "model.onnx.data").touch()
            (model_directory / "model_vocabulary.json").touch()

            model_path, vocabulary_path = resolve_model_and_labels(
                "cl-tagger.v2", directory
            )

        self.assertEqual(str(model_directory / "model.onnx"), model_path)
        self.assertEqual(
            str(model_directory / "model_vocabulary.json"), vocabulary_path
        )

    def test_tag_image_uses_siglip_preprocessing_and_preserves_response_groups(self):
        logits = [
            math.log(0.8 / 0.2),
            math.log(0.2 / 0.8),
            math.log(0.9 / 0.1),
            0.0,
            math.log(0.6 / 0.4),
        ]
        model = _FakeSession(logits)
        labels = (
            ["general", "explicit", "1girl", "low_score", "hatsune_miku"],
            [0, 1],
            [2, 3],
            [4],
        )
        image = Image.new("RGB", (4, 2), (255, 0, 0))

        result = tag_image(image, model, labels)

        self.assertEqual((1, 3, 2, 2), model.received_input.shape)
        np.testing.assert_allclose(model.received_input[:, 0], 1.0)
        np.testing.assert_allclose(model.received_input[:, 1:], -1.0)
        self.assertAlmostEqual(0.8, result["rating"]["general"], places=6)
        self.assertAlmostEqual(0.2, result["rating"]["explicit"], places=6)
        self.assertEqual(["1girl"], list(result["tags"]))
        self.assertEqual(["hatsune miku"], list(result["characters"]))

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


if __name__ == "__main__":
    unittest.main()

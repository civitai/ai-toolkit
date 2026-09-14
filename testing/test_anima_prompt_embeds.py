import unittest
from types import SimpleNamespace

import torch
from diffusers import AnimaTextConditioner
from transformers import Qwen3Config, Qwen3Model

from extensions_built_in.diffusion_models.anima.anima import AnimaModel


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, prompts, **kwargs):
        input_ids = torch.tensor([[1, 2, 3]] * len(prompts))
        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
        )


def _model(device, gradient_checkpointing=True):
    model = AnimaModel.__new__(AnimaModel)
    model.device_torch = torch.device(device)
    model.torch_dtype = torch.float32
    model.max_sequence_length = 16
    model.model_config = SimpleNamespace(low_vram=True)
    encoder = Qwen3Model(Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    ))
    encoder.requires_grad_(False)
    if gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    # The trainer sets training mode when train_text_encoder is requested,
    # even though Anima's Qwen encoder has no trainable LoRA modules.
    encoder.train()
    model.pipeline = SimpleNamespace(text_encoder=encoder, tokenizer=_Tokenizer())
    model.t5_tokenizer = _Tokenizer()
    return model


class AnimaPromptEmbedsTests(unittest.TestCase):
    def test_frozen_encoder_does_not_build_a_backward_graph(self):
        for checkpointing in (False, True):
            with self.subTest(gradient_checkpointing=checkpointing):
                model = _model('cpu', gradient_checkpointing=checkpointing)
                with torch.enable_grad():
                    embeds = model.get_prompt_embeds(['test'])
                self.assertFalse(embeds.text_embeds.requires_grad)
                self.assertIsNone(embeds.text_embeds.grad_fn)
                self.assertTrue(torch.isfinite(embeds.text_embeds).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_offloaded_encoder_preserves_text_conditioner_gradients(self):
        model = _model('cuda:0')
        conditioner = AnimaTextConditioner(
            source_dim=32,
            target_dim=32,
            model_dim=32,
            num_layers=1,
            num_attention_heads=4,
            target_vocab_size=32,
            min_sequence_length=3,
        ).to(model.device_torch)
        model.model = SimpleNamespace(
            text_conditioner=conditioner,
            transformer=SimpleNamespace(dtype=torch.float32),
        )
        for _ in range(2):
            conditioner.zero_grad(set_to_none=True)
            embeds = model.get_prompt_embeds(['test'])
            self.assertEqual(model.pipeline.text_encoder.device, torch.device('cpu'))
            self.assertEqual(embeds.text_embeds.device, model.device_torch)
            conditioned = model._condition_prompt_embeds(embeds)
            conditioned.square().mean().backward()
            grads = [p.grad for p in conditioner.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
            self.assertTrue(any(torch.count_nonzero(grad) for grad in grads))
            self.assertTrue(all(p.grad is None for p in model.pipeline.text_encoder.parameters()))


if __name__ == '__main__':
    unittest.main()

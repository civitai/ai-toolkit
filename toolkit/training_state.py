"""Portable, weights-only training checkpoints paired with an exported LoRA."""

import hashlib
import json
import os
import random
import tempfile
import zipfile

import numpy as np
import torch


VERSION = 1


def file_hash(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def _optimizer(trainer):
    optimizer = trainer.optimizer
    # Accelerate wraps the optimizer after setup; use the same type on either worker.
    from accelerate.optimizer import AcceleratedOptimizer
    while isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer
    return optimizer


def _type_name(value):
    return f'{type(value).__module__}.{type(value).__qualname__}'


def save_training_state(trainer, checkpoint_path, step):
    optimizer = _optimizer(trainer)
    numpy_state = np.random.get_state()
    scaler = getattr(trainer.accelerator, 'scaler', None)
    state = {
        'optimizer_type': _type_name(optimizer),
        'optimizer': optimizer.state_dict(),
        # The exported LoRA can be bf16 or EMA weights. Resume the actual training parameters.
        'parameters': [[p.detach() for p in group['params']] for group in optimizer.param_groups],
        'gradients': [[p.grad for p in group['params']] for group in optimizer.param_groups],
        'scheduler_type': _type_name(trainer.lr_scheduler),
        'scheduler': trainer.lr_scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'ema': trainer.ema.state_dict() if trainer.ema is not None else None,
        'step': step,
        'epoch': trainer.epoch_num,
        'grad_accumulation_step': trainer.grad_accumulation_step,
        'python_rng': random.getstate(),
        'numpy_rng': (numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)), *numpy_state[2:]),
        'torch_rng': torch.get_rng_state(),
        # Controller sessions use one visible GPU. Restore onto the new worker's current device.
        'cuda_rng': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    manifest = {
        'version': VERSION,
        'step': step,
        'checkpoint_sha256': file_hash(checkpoint_path),
    }
    destination = os.path.splitext(checkpoint_path)[0] + '.training-state.zip'
    # Never publish a partially written state or report the pause before both files exist.
    with tempfile.TemporaryDirectory(dir=trainer.save_root, prefix='.training-state-') as temp_dir:
        state_path = os.path.join(temp_dir, 'state.pt')
        torch.save(_cpu(state), state_path)
        manifest['state_sha256'] = file_hash(state_path)
        archive_path = os.path.join(temp_dir, 'state.zip')
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.write(state_path, 'state.pt')
            archive.writestr('manifest.json', json.dumps(manifest))
        os.replace(archive_path, destination)
    return destination


def load_training_state(trainer, state_path, checkpoint_path):
    # Do not fall back to fresh optimization on a corrupt or incompatible resume.
    with zipfile.ZipFile(state_path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['version'] != VERSION or manifest['step'] != trainer.step_num:
            raise ValueError('Training state version or step does not match the requested checkpoint')
        if manifest['checkpoint_sha256'] != file_hash(checkpoint_path):
            raise ValueError('Training state belongs to a different LoRA checkpoint')
        with archive.open('state.pt') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest['state_sha256']:
                raise ValueError('Training state checksum mismatch')
        with archive.open('state.pt') as stream:
            state = torch.load(stream, map_location='cpu', weights_only=True)

    optimizer = _optimizer(trainer)
    if state['step'] != trainer.step_num or state['optimizer_type'] != _type_name(optimizer):
        raise ValueError('Training state step or optimizer does not match the session')
    if state['scheduler_type'] != _type_name(trainer.lr_scheduler):
        raise ValueError('Training state scheduler does not match the session')
    if len(state['parameters']) != len(optimizer.param_groups):
        raise ValueError('Training state parameter groups do not match the session')
    with torch.no_grad():
        for group, parameters, gradients in zip(optimizer.param_groups, state['parameters'], state['gradients'], strict=True):
            for param, saved, gradient in zip(group['params'], parameters, gradients, strict=True):
                if param.shape != saved.shape or param.dtype != saved.dtype:
                    raise ValueError('Training state parameter shape or precision does not match')
                param.copy_(saved)
                param.grad = gradient.to(param.device) if gradient is not None else None
    optimizer.load_state_dict(state['optimizer'])
    trainer.lr_scheduler.load_state_dict(state['scheduler'])
    scaler = getattr(trainer.accelerator, 'scaler', None)
    for name, target in [('scaler', scaler), ('ema', trainer.ema)]:
        if (state[name] is None) != (target is None):
            raise ValueError(f'Training state {name} does not match the session')
        if target is not None:
            target.load_state_dict(state[name])
    trainer.epoch_num = state['epoch']
    trainer.grad_accumulation_step = state['grad_accumulation_step']
    random.setstate(state['python_rng'])
    numpy_state = state['numpy_rng']
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA training state requires a CUDA device')
        torch.cuda.set_rng_state(state['cuda_rng'])


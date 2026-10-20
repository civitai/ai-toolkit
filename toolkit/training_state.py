"""Portable, weights-only training checkpoints paired with an exported LoRA."""

import hashlib
import json
import os
import random
import tempfile
import zipfile

import numpy as np
import torch


VERSION = 2


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


def _parameter_layout(parameters):
    return [[{'shape': list(p.shape), 'dtype': str(p.dtype)} for p in group]
            for group in parameters]


def _can_restore_from_lora(trainer, parameters):
    # EMA exports are not the live weights. Other trainable modules may not be
    # included in the paired LoRA either. Keep their original full-state path.
    network = trainer.network
    if network is None or trainer.ema is not None:
        return False
    network_parameters = {id(p) for p in network.parameters()}
    return all(id(p) in network_parameters for group in parameters for p in group)


def save_training_state(trainer, checkpoint_path, step):
    optimizer = _optimizer(trainer)
    parameters = [list(group['params']) for group in optimizer.param_groups]
    numpy_state = np.random.get_state()
    scaler = getattr(trainer.accelerator, 'scaler', None)
    state = {
        'optimizer_type': _type_name(optimizer),
        'optimizer': optimizer.state_dict(),
        # Preserve optimizer layout validation without duplicating the LoRA's weights.
        'parameter_layout': _parameter_layout(parameters),
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
    if not _can_restore_from_lora(trainer, parameters):
        state['parameters'] = parameters
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


def load_training_state(trainer, state_path, checkpoint_path, mode='resume'):
    if mode not in ('resume', 'continue'):
        raise ValueError('Unknown training state restore mode')
    continuing = mode == 'continue'
    if continuing and trainer.step_num != 0:
        raise ValueError('A new continuation must start at step zero')
    # Corrupt or mismatched artifacts are errors, even for a new continuation.
    with zipfile.ZipFile(state_path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['version'] not in (1, VERSION) or (not continuing and manifest['step'] != trainer.step_num):
            raise ValueError('Training state version or step does not match the requested checkpoint')
        if manifest['checkpoint_sha256'] != file_hash(checkpoint_path):
            raise ValueError('Training state belongs to a different LoRA checkpoint')
        with archive.open('state.pt') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != manifest['state_sha256']:
                raise ValueError('Training state checksum mismatch')
        with archive.open('state.pt') as stream:
            state = torch.load(stream, map_location='cpu', weights_only=True)

    optimizer = _optimizer(trainer)
    current_parameters = [list(group['params']) for group in optimizer.param_groups]
    parameters = state.get('parameters')
    layout = (state['parameter_layout'] if manifest['version'] == VERSION
              else _parameter_layout(state['parameters']))
    if state['step'] != manifest['step']:
        raise ValueError('Training state step does not match its manifest')
    incompatible = None
    if state['optimizer_type'] != _type_name(optimizer):
        incompatible = 'optimizer type'
    elif len(layout) != len(optimizer.param_groups):
        incompatible = 'parameter groups'
    elif layout != _parameter_layout(current_parameters):
        incompatible = 'parameter shape or precision'
    elif parameters is None and not _can_restore_from_lora(trainer, current_parameters):
        incompatible = 'LoRA parameter coverage'
    if incompatible is not None:
        if continuing:
            return f'weights only (incompatible {incompatible})'
        raise ValueError(f'Training state {incompatible} does not match the session')
    if not continuing and state['scheduler_type'] != _type_name(trainer.lr_scheduler):
        raise ValueError('Training state scheduler does not match the session')
    # Check stored tensor layouts before mutating the network or optimizer.
    if parameters is not None and _parameter_layout(parameters) != layout:
        raise ValueError('Training state parameters do not match their layout')
    if len(state['gradients']) != len(current_parameters) or any(
        len(gradients) != len(group) or any(
            gradient is not None and (gradient.shape != p.shape or gradient.dtype != p.dtype)
            for p, gradient in zip(group, gradients)
        ) for group, gradients in zip(current_parameters, state['gradients'])
    ):
        raise ValueError('Training state gradients do not match the session')
    learning_rates = [{key: group[key] for key in ('lr', 'initial_lr') if key in group}
                     for group in optimizer.param_groups]
    if parameters is None:
        # Reload the verified pair explicitly, never rely on a previously loaded
        # checkpoint. FP16/BF16 export rounding is intentionally retained on resume.
        trainer.network.load_weights(checkpoint_path)
    with torch.no_grad():
        for index, (group, gradients) in enumerate(zip(current_parameters, state['gradients'], strict=True)):
            for offset, (param, gradient) in enumerate(zip(group, gradients, strict=True)):
                if parameters is not None:
                    param.copy_(parameters[index][offset])
                param.grad = gradient.to(param.device) if gradient is not None and not continuing else None
    optimizer.load_state_dict(state['optimizer'])
    if continuing:
        # Preserve moments and optimizer counters, but use the new run's LR and
        # fresh scheduler, progress, RNG, gradient accumulation and scaler.
        for group, rates in zip(optimizer.param_groups, learning_rates, strict=True):
            group.update(rates)
        if trainer.ema is not None:
            ema_state = trainer.ema.state_dict()
            ema_state['shadow_params'] = [p.detach().clone() for group in optimizer.param_groups for p in group['params']]
            trainer.ema.load_state_dict(ema_state)
        return 'optimizer (new learning-rate schedule and step budget)'
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
    return 'optimizer and training state (LoRA export weights)' if parameters is None else 'full training state'


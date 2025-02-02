# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
import torch

from nemo.collections.tts.data.text_to_speech_dataset import T5TTSDataset, DatasetSample
from nemo.collections.tts.models import T5TTS_Model, T5TTS_ModelInference, T5TTS_ModelDPO
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

def json_reader(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)

def get_data_samples(filename):
    test_entries = list(json_reader(filename))
    audio_base_dir = "/"
    data_samples = []
    for entry in test_entries:
        dataset_sample = DatasetSample(
            dataset_name="sample",
            manifest_entry=entry,
            audio_dir=audio_base_dir,
            feature_dir=audio_base_dir,
            text=entry['text'],
            speaker=None,
            speaker_index=0
        )
        data_samples.append(dataset_sample)
    return data_samples

@hydra_runner(config_path="conf/t5tts", config_name="t5tts")
def main(cfg):
    logging.info('\nConfig Params:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    if not cfg.model.get('use_lthose', False):
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

    # Inference is done on multiple manifests (test_ds_paths) and the output is saved in shar_output_paths
    test_ds_paths = cfg.test_ds_paths
    shar_output_paths = cfg.shar_output_paths

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.get('mode', 'train') == 'train':
        model = T5TTS_Model(cfg=cfg.model, trainer=trainer)
    elif cfg.get('mode', 'dpo_train') == 'dpo_train':
        model_cfg = cfg.model
        with open_dict(model_cfg):
            model_cfg.reference_model_ckpt_path = cfg.init_from_ptl_ckpt
        model = T5TTS_ModelDPO(cfg=model_cfg, trainer=trainer)
    elif cfg.get('mode', 'train') == 'test':
        model = T5TTS_ModelInference(cfg=cfg.model, trainer=trainer)
    else:
        raise NotImplementedError(f"Only train, dpo_train and test modes are supported. Got {cfg.mode}")

    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    
    if cfg.get('mode', 'train') in ['train', 'dpo_train']:
        trainer.fit(model)
    elif cfg.get('mode', 'train') == 'test':
        trainer.test(model)
    else:
        raise NotImplementedError(f"Only train and test modes are supported. Got {cfg.mode}")
    print(f"DONE WITH DUMMY TEST")
    
    world_size = torch.cuda.device_count()
    rank = torch.distributed.get_rank()

    for ind, (currect_test_ds, shar_output_path) in enumerate(zip(test_ds_paths, shar_output_paths)):
        print(f"...{rank} TEST {ind}")
        model._test_dl.dataset.text_tokenizer, model._test_dl.dataset.text_conditioning_tokenizer = model._setup_tokenizers(model.cfg, mode='test')
        test_data_samples = get_data_samples(currect_test_ds)
        model._test_dl.dataset.data_samples = []
        model._test_dl.dataset.data_samples = test_data_samples

        sampler = torch.utils.data.distributed.DistributedSampler(
            model._test_dl.dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=1
        )

        model._test_dl = torch.utils.data.DataLoader(
            model._test_dl.dataset,
            collate_fn=model._test_dl.dataset.collate_fn,
            sampler=sampler,
            batch_size=int(cfg.model.test_ds.dataloader_params.batch_size),
            drop_last=False,
            num_workers=cfg.model.test_ds.dataloader_params.num_workers,
            pin_memory=True,
            persistent_workers=False
        )

        model.save_in_lhotse_shars = cfg.model.save_in_lhotse_shars
        model.shar_output_path = shar_output_path
        # model.cfg.data.test_ds = None
        model.predict_step_outputs = []
        trainer.test(model, model._test_dl)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

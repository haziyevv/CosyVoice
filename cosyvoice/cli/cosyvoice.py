# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from typing import Generator
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type
from cosyvoice.utils.common import ras_sampling
import torch
import functools
import pickle

# from cosyvoice.llm_wrapper import Qwen2LMOnnxWrapper
import random
import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn


TRACED_LLM_EMBEDDING_PATH = "llm_embedding_fp32_tracing.pt"
TRACED_SPEECH_EMBEDDING_PATH = "speech_embedding_fp32_tracing.pt"


class Qwen2LMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.llm_decoder = model.llm_decoder
    
    def tensor_to_tuple_cache(self, cache_tensor):
        """Convert single tensor [L,2,B,H,S,D] to tuple of (k,v) expected by model."""
        L = cache_tensor.shape[0]
        cache_tuple = []
        for i in range(L):
            # From [L,2,B,H,S,D] -> [B,H,S,D]
            k = cache_tensor[i, 0]  # Already in [B,H,S,D]
            v = cache_tensor[i, 1]  # Already in [B,H,S,D]
            cache_tuple.append((k, v))
        return tuple(cache_tuple)

    def forward(self, lm_input, cache):
        cache = self.tensor_to_tuple_cache(cache)
        y_pred, new_cache = self.model.llm.forward_one_step(
            lm_input,
            masks=torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool),
            cache=cache
        )
        logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
        return logp, new_cache


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_traced_model(model_dir):
    """Create and save traced versions of both embedder and LLM"""
    # Load model as before

    def set_seeds(seed=1986):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    # Initialize models
    set_seeds()
    with open(f'{model_dir}/cosyvoice2.yaml', 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
    
    model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], False, False)
    model.load(f'{model_dir}/llm.pt',
              f'{model_dir}/flow.pt',
              f'{model_dir}/hift.pt')
    
    # # 1. Trace embedder
   #embedder = model.llm.llm.model.model.embed_tokens.eval().cuda()

    # dummy_text = torch.randint(0, 32000, (1, 60)).cuda()  # Adjust to match cache size
    # traced_embedder = torch.jit.trace(embedder, (dummy_text,))
    
    # 2. Trace LLM (post-embedding)
    wrapped = Qwen2LMWrapper(model.llm).eval().cuda()
    # dummy_lm_input = traced_embedder(dummy_text)  # Use output from traced embedder
    
    with open('cache_file.pkl', 'rb') as fr:
        dummy_cache = pickle.load(fr)
    with open('lm_input_file.pkl', 'rb') as fr:
        dummy_lm_input = pickle.load(fr)
    
    # Trace the model
    dummy_cache = tuple_to_tensor_cache(dummy_cache)
    traced_llm = torch.jit.trace(wrapped, (dummy_lm_input, dummy_cache))
    traced_llm = torch.jit.optimize_for_inference(traced_llm)
    # Save both models
    traced_embedder_path = f"{model_dir}/qwen2_embedder_traced.pt"
    traced_llm_path = f"{model_dir}/qwen2_llm_traced.pt"
    
    #traced_embedder.save(traced_embedder_path)
    traced_llm.save(traced_llm_path)
    
    print(f"Traced models saved to {traced_embedder_path} and {traced_llm_path}")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    return traced_embedder_path, traced_llm_path

class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False, use_traced_llm=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = '{}/cosyvoice2.yaml'.format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError('{} not found!'.format(hyper_yaml_path))
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16, use_flow_cache)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir) if use_flow_cache is False else '{}/flow.cache.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))

        self.model.sampling = functools.partial(ras_sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1)
        
        # Load the traced LLM model if it exists
        traced_path = f"{model_dir}/qwen2_llm_traced.pt"
        embedder_traced_path = f"{model_dir}/qwen2_embedder_traced.pt"
        if use_traced_llm and os.path.exists(traced_path):
            logging.info('Loading traced LLM model from {}'.format(traced_path))
            self.model.llm = torch.jit.load(traced_path).to(device)

            self.model.qwen2_embedder_traced = torch.jit.load(embedder_traced_path).to(device)
            self.model.traced_llm_embedding = torch.load(TRACED_LLM_EMBEDDING_PATH, map_location="cuda")
            self.model.traced_speech_embedding = torch.load(TRACED_SPEECH_EMBEDDING_PATH, map_location="cuda")
        # Tensor, not Parameter
        else:
            logging.warning('Traced model not found at {}. Using original model.'.format(traced_path))
            self.model.llm.to(device)

        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                self.fp16)
        del configs

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

from __future__ import annotations

import logging
import os
import warnings

import torch

from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from ...smp import get_gpu_memory, listinstr


VLLM_MAX_IMAGE_INPUT_NUM = 128


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a MoE model by looking for active-param suffixes like A3B, A17B."""
    import re
    if re.search(r'-A\d+B', model_path):
        return True
    return False

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        use_cot: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.do_sample = kwargs.pop('do_sample', self.temperature > 0)
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
        self.generate_kwargs = dict(
            do_sample=self.do_sample,
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process

        # USE_COT env var allows runtime override without changing config entries.
        use_cot = use_cot or os.environ.get('USE_COT', '0') == '1'
        if use_cot:
            self.temperature = 0.7
            self.do_sample = True
            self.max_new_tokens = 2048
            self.post_prompt = (
                'Please think step by step inside <think> tags, '
                'then provide the final answer inside <answer> tags.'
            )
            self.extract_think_answer = True
            self.generate_kwargs.update(
                temperature=0.7, do_sample=True, max_new_tokens=2048
            )
        else:
            self.post_prompt = None
            self.extract_think_answer = False

        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()

            # Tensor Parallelism has diminishing returns for small models due to
            # all-reduce communication overhead. Cap TP size based on model size:
            #   < 8B  params  → tp=1 (single GPU, zero comm overhead)
            #   8B–30B params → tp=2
            #   > 30B params  → use all available GPUs
            # Users can override via VLLM_TP_SIZE env var.
            def _default_tp_size(model_path: str, available: int) -> int:
                import re
                m = re.search(r'[-_](\d+(?:\.\d+)?)B', model_path, re.IGNORECASE)
                if m:
                    params_b = float(m.group(1))
                    if params_b < 8:
                        return 1
                    elif params_b < 30:
                        return min(2, available)
                return available if available > 0 else 1

            env_tp = os.environ.get('VLLM_TP_SIZE', '')
            if env_tp.isdigit():
                tp_size = min(int(env_tp), gpu_count)
            else:
                tp_size = _default_tp_size(self.model_path, gpu_count)

            logging.info(
                f'Using vLLM for {self.model_path} inference with tp_size={tp_size} '
                f'(available GPUs: {gpu_count}). Set VLLM_TP_SIZE env var to override.'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            max_num_seqs = int(os.environ.get('VLLM_MAX_NUM_SEQS', '8'))

            # ── Isolate vLLM from torchrun's distributed env vars ──
            # torchrun sets MASTER_ADDR, MASTER_PORT, RANK, LOCAL_RANK, etc.
            # vLLM's EngineCore subprocess inherits these and mistakenly tries
            # to connect to torchrun's TCPStore, causing a timeout.  We strip
            # them before creating the LLM instance and restore afterwards.
            _dist_env_keys = [
                'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'LOCAL_RANK',
                'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'TORCHELASTIC_RUN_ID',
            ]
            _dist_env_backup = {k: os.environ.pop(k) for k in _dist_env_keys if k in os.environ}

            # Destroy the existing torch.distributed process group (gloo)
            # so vLLM's EngineCore subprocess doesn't inherit stale state.
            import torch.distributed as _dist
            _had_dist = _dist.is_initialized()
            if _had_dist:
                logging.info('[vLLM init] Destroying existing torch.distributed process group before LLM()')
                _dist.destroy_process_group()

            # Force vLLM to bind to loopback
            os.environ['VLLM_HOST_IP'] = '127.0.0.1'

            logging.info(
                f'[vLLM init] Temporarily removed torchrun env vars: {list(_dist_env_backup.keys())}'
            )
            try:
                self.llm = LLM(
                    model=self.model_path,
                    max_num_seqs=max_num_seqs,
                    # limit_mm_per_prompt=limit_mm,
                    tensor_parallel_size=tp_size,
                    enable_expert_parallel=enable_expert_parallel,
                    seed=0,
                    max_model_len=32768,  
                    enforce_eager=True,
                    gpu_memory_utilization=kwargs.get("gpu_utils", float(os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', 0.85))),
                    trust_remote_code=True,
                )
            finally:
                os.environ.update(_dist_env_backup)
                # Re-initialize the process group so barrier() calls still work.
                if _had_dist:
                    logging.info('[vLLM init] Re-initializing gloo process group after LLM()')
                    import datetime as _dt
                    _dist.init_process_group(
                        backend='gloo',
                        timeout=_dt.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
                    )
                    logging.info(f'[vLLM init] gloo process group re-initialized successfully')
                logging.info(
                    f'[vLLM init] Restored torchrun env vars: {list(_dist_env_backup.keys())}'
                )
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
                if 'role' in s:
                    item['role'] = s['role']
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        if self.post_prompt:
            content = self._rewrite_prompt_for_cot(content)
        return content

    # Patterns that instruct the model to answer directly (no CoT).
    # Used by _rewrite_prompt_for_cot to strip them before appending a CoT instruction.
    _DIRECT_ANSWER_PATTERNS = [
        r'\n?Only give the best option\.?',
        r'\n?Answer with the option letter only\.?',
        r'\n?Answer with the option\'?s letter from the given choices directly\.?',
        r'Respond with only the letter \([A-F][^)]*\) of the correct option\.?\s?',
    ]

    def _rewrite_prompt_for_cot(self, content: list[dict]) -> list[dict]:
        """Strip 'answer directly' instructions from dataset prompts and append CoT instruction.

        Also removes assistant-role prefill messages (e.g. MVBench 'Best option:(')
        since CoT output won't start with an option letter.
        """
        import re
        pattern = '|'.join(self._DIRECT_ANSWER_PATTERNS)
        rewritten = []
        for item in content:
            # Drop assistant prefill messages (MVBench)
            if item.get('type') == 'text' and item.get('role') == 'assistant':
                continue
            if item.get('type') == 'text':
                text = re.sub(pattern, '', item['text']).strip()
                if text:
                    rewritten.append({**item, 'text': text})
            else:
                rewritten.append(item)
        rewritten.append({'type': 'text', 'text': self.post_prompt})
        return rewritten

    def generate_inner_transformers(self, message, dataset=None):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if is_omni:
            # For Qwen3-Omni, messages is a list of dicts
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')

        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            generated_ids = self.model.generate(
                **inputs,
                **self.generate_kwargs,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.extract_think_answer:
            import re
            m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if m:
                response = m.group(1).strip()
            else:
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response

    def _build_vllm_request(self, message, dataset=None):
        """Build a single vLLM request dict from a message. Used for both single and batch inference."""
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        return req

    def generate_batch_vllm(self, messages, dataset=None, chunk_size=None):
        """Batch inference for a list of messages. Returns a list of response strings.

        Processes in chunks to avoid building all multimodal inputs in memory at once
        (1000+ video samples * 16 frames would stall before generation even starts).
        """
        if chunk_size is None:
            chunk_size = int(os.environ.get('VLLM_BATCH_CHUNK_SIZE', '32'))
        from vllm import SamplingParams
        from tqdm import tqdm as _tqdm
        sampling_params = SamplingParams(
            temperature=self.temperature if self.do_sample else 0.0,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p if self.do_sample else 1.0,
            top_k=self.top_k if self.do_sample else -1,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None,
        )
        results = []
        total_chunks = (len(messages) + chunk_size - 1) // chunk_size
        for chunk_start in _tqdm(range(0, len(messages), chunk_size),
                                 total=total_chunks,
                                 desc='vLLM batch generate (chunks)'):
            chunk = messages[chunk_start: chunk_start + chunk_size]
            reqs = [self._build_vllm_request(msg, dataset=dataset) for msg in chunk]
            outputs = self.llm.generate(reqs, sampling_params=sampling_params, use_tqdm=False)
            for o in outputs:
                generated_text = o.outputs[0].text
                if self.post_process:
                    resp = generated_text.split('\\boxed{')[-1]
                    lt = len(resp)
                    counter, end = 1, None
                    for i in range(lt):
                        if resp[i] == '{':
                            counter += 1
                        elif resp[i] == '}':
                            counter -= 1
                        if counter == 0:
                            end = i
                            break
                        elif i == lt - 1:
                            end = lt
                            break
                    if end is not None:
                        generated_text = resp[:end]
                if self.extract_think_answer:
                    import re
                    m = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
                    if m:
                        generated_text = m.group(1).strip()
                    else:
                        generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()
                results.append(generated_text)
        return results

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        if self.verbose:
            messages_preview = []
            if self.system_prompt is not None:
                messages_preview.append({'role': 'system', 'content': self.system_prompt})
            messages_preview.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
            print(f'\033[31m{messages_preview}\033[0m')

        sampling_params = SamplingParams(
            temperature=self.temperature if self.do_sample else 0.0,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p if self.do_sample else 1.0,
            top_k=self.top_k if self.do_sample else -1,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None
        )
        req = self._build_vllm_request(message, dataset=dataset)
        outputs = self.llm.generate([req], sampling_params=sampling_params, use_tqdm=False)

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.extract_think_answer:
            import re
            m = re.search(r'<answer>(.*?)</answer>', generated_text, re.DOTALL)
            if m:
                generated_text = m.group(1).strip()
            else:
                generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)

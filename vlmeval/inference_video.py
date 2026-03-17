import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    if getattr(model,'backend', None) == 'genai':
        if dataset.nframe > 0:
            print(
                'Gemini model (with genai backend) does not support nframe, '
                'will set its VIDEO_LLM to False to enable multi-image input for video.'
            )
            setattr(model, 'VIDEO_LLM', False)
        else:
            print('Gemini model (with genai backend) is a video-llm, '
                  'will reset fps setting in model to match the dataset.')
            setattr(model, 'fps', dataset.fps)
            print(f'The fps is set to {dataset.fps} for the model {model_name}.')
    elif getattr(model,'backend', None) == 'vertex':
        print('Gemini model (with vertex backend) does not support video input, '
              'will set its VIDEO_LLM to False to enable multi-image input for video.')
        setattr(model, 'VIDEO_LLM', False)

    packstr = 'pack' if getattr(dataset, 'pack', False) else 'nopack'
    build_prompt_input = [(samples_dict[idx], getattr(model, 'VIDEO_LLM', False)) for idx in indices]
    if dataset.nframe > 0:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_structs.pkl'
    else:
        struct_tmp_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_structs.pkl'
    structs = track_progress_rich(
        dataset.build_prompt,
        tasks=build_prompt_input,
        nproc=api_nproc,
        save=struct_tmp_file,
        keys=indices,
    )

    if dataset.nframe > 0:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.nframe}frame_{packstr}_supp.pkl'
    else:
        out_file = f'{work_dir}/{model_name}_{dataset_name}_{dataset.fps}fps_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res or res[i] == FAIL_MSG]
    structs = [struct for struct in structs if struct is not None]
    indices = [i for i in indices if i not in res or res[i] == FAIL_MSG]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    skip_err = os.environ.get('SKIP_ERR', '0') == '1'

    sample_indices = list(dataset.videos) if getattr(dataset, 'pack', False) else list(dataset.data['index'])
    samples = list(dataset.videos) if getattr(dataset, 'pack', False) else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    if np.all([idx in res for idx in sample_indices_sub]):
        return model
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
        or 'Qwen2.5-Omni' in model_name
        or 'Qwen3-VL' in model_name
        or 'Qwen3-Omni' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak
    
    # =============== [开始] 插入保存配置代码 ===============
    # 只有 rank 0 (主进程) 负责保存，避免多进程写冲突
    if rank == 0:
        try:
            import json
            # work_dir 就是结果输出目录
            config_save_path = osp.join(work_dir, 'model_config.json')
            
            # 只有文件不存在时才保存，避免重复覆盖
            if not osp.exists(config_save_path):
                saved_config = {
                    "model_name": model_name,
                    "model_class": type(model).__name__,
                    # 使用 getattr 安全获取属性，防止报错
                    "model_path": getattr(model, 'model_path', 'Unknown'),
                    "use_vllm": getattr(model, 'use_vllm', False),
                    "use_lmdeploy": getattr(model, 'use_lmdeploy', False),
                    "system_prompt": getattr(model, 'system_prompt', None),
                    "generation_kwargs": getattr(model, 'generate_kwargs', {}),
                    "min_pixels": getattr(model, 'min_pixels', None),
                    "max_pixels": getattr(model, 'max_pixels', None),
                    "total_pixels": getattr(model, 'total_pixels', None),
                    "fps": getattr(model, 'fps', None),
                    "nframe": getattr(model, 'nframe', None),
                    # 特别保存 limit_mm_per_prompt，这对应你修改的 VLLM_MAX_IMAGE_INPUT_NUM
                    "limit_mm_per_prompt": getattr(model, 'limit_mm_per_prompt', None)
                }
                
                with open(config_save_path, 'w', encoding='utf-8') as f:
                    json.dump(saved_config, f, indent=4, ensure_ascii=False)
                print(f"[Config] Model configuration saved to: {config_save_path}")
        except Exception as e:
            print(f"[Config] Warning: Failed to save model config. Error: {e}")
    # =============== [结束] 插入保存配置代码 ===============

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model

    assert not getattr(dataset, 'pack', False), 'Current model not supported pack mode!'
    if 'megabench' in dataset_name.lower() and 'llava_onevision' in model_name:
        print(
            'LLaVA-OneVision does not support Megabench dataset as video dataset, '
            'will set its VIDEO_LLM to False to enable multi-image input for video.'
        )
        setattr(model, 'VIDEO_LLM', False)

    # ------------------------------------------------------------------
    # vLLM batch path: collect all prompts first, then run a single
    # batched llm.generate() call to saturate all tensor-parallel GPUs.
    # ------------------------------------------------------------------
    if getattr(model, 'use_vllm', False) and hasattr(model, 'generate_batch_vllm'):
        # Sync nframe / fps once before building prompts
        if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
            if dataset.nframe > 0 and getattr(model, 'nframe', 0) != dataset.nframe:
                print(f'{model_name} nframe -> {dataset.nframe}')
                setattr(model, 'nframe', dataset.nframe)
        if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
            if dataset.fps > 0 and getattr(model, 'fps', 0) != dataset.fps:
                print(f'{model_name} fps -> {dataset.fps}')
                setattr(model, 'fps', dataset.fps)

        batch_indices, batch_structs = [], []
        for idx in tqdm(sample_indices_subrem, desc=f'Build prompts {model_name}/{dataset_name}'):
            if idx in res:
                continue
            try:
                if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
                    struct = model.build_prompt(
                        dataset.data.iloc[sample_map[idx]], dataset=dataset,
                        video_llm=getattr(model, 'VIDEO_LLM', False)
                    )
                else:
                    struct = dataset.build_prompt(sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False))
            except Exception as err:
                if not skip_err:
                    raise
                logging.warning(
                    f'Skip sample {idx} in {model_name}/{dataset_name} during prompt build: '
                    f'{type(err).__name__}: {err}'
                )
                res[idx] = ''
                continue
            if struct is None:
                continue
            batch_indices.append(idx)
            batch_structs.append(struct)

        if batch_structs:
            chunk_size = 32
            for chunk_start in range(0, len(batch_structs), chunk_size):
                chunk_indices = batch_indices[chunk_start: chunk_start + chunk_size]
                chunk_structs = batch_structs[chunk_start: chunk_start + chunk_size]
                responses = model.generate_batch_vllm(chunk_structs, dataset=dataset_name, chunk_size=chunk_size)
                for idx, resp in zip(chunk_indices, responses):
                    res[idx] = resp
                dump(res, out_file)

        res = {k: res[k] for k in sample_indices_sub}
        dump(res, out_file)
        return model
    # ------------------------------------------------------------------

    for i, idx in enumerate(
        tqdm(
            sample_indices_subrem,
            total=len(sample_indices_subrem),
            desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}',
        )
    ):
        if idx in res:
            continue
        if getattr(model, 'nframe', None) is not None and getattr(model, 'nframe', 0) > 0:
            if dataset.nframe > 0:
                if getattr(model, 'nframe', 0) != dataset.nframe:
                    print(f'{model_name} is a video-llm model, nframe is set to {dataset.nframe}, not using default')
                    setattr(model, 'nframe', dataset.nframe)
            elif getattr(model, 'fps', 0) == 0:
                raise ValueError(f'fps is not suitable for {model_name}')
            else:
                setattr(model, 'nframe', None)
        if getattr(model, 'fps', None) is not None and getattr(model, 'fps', 0) > 0:
            if dataset.fps > 0:
                if getattr(model, 'fps', 0) != dataset.fps:
                    print(f'{model_name} is a video-llm model, fps is set to {dataset.fps}, not using default')
                    setattr(model, 'fps', dataset.fps)
            elif getattr(model, 'nframe', 0) == 0:
                raise ValueError(f'nframe is not suitable for {model_name}')
            else:
                setattr(model, 'fps', None)
        if (
            'Qwen2-VL' in model_name
            or 'Qwen2.5-VL' in model_name
            or 'Qwen2.5-Omni' in model_name
        ):
            if getattr(model, 'nframe', None) is None and dataset.nframe > 0:
                print(f'using {model_name} default setting for video, dataset.nframe is ommitted')
            if getattr(model, 'fps', None) is None and dataset.fps > 0:
                print(f'using {model_name} default setting for video, dataset.fps is ommitted')
        if 'SUB_DATASET' in dataset.data.iloc[sample_map[idx]]:
            dataset_name = dataset.data.iloc[sample_map[idx]]['SUB_DATASET']
        try:
            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
                if dataset.nframe == 0:
                    raise ValueError(f'nframe must be set for custom prompt, fps is not suitable for {model_name}')
                struct = model.build_prompt(
                    dataset.data.iloc[sample_map[idx]], dataset=dataset, video_llm=getattr(model, 'VIDEO_LLM', False)
                )
            else:
                struct = dataset.build_prompt(
                    sample_map[idx], video_llm=getattr(model, 'VIDEO_LLM', False)
                )
        except Exception as err:
            if not skip_err:
                raise
            logging.warning(
                f'Skip sample {idx} in {model_name}/{dataset_name} during prompt build: '
                f'{type(err).__name__}: {err}'
            )
            res[idx] = ''
            if (i + 1) % 20 == 0:
                dump(res, out_file)
            continue
        if struct is None:
            continue

        # Print the first batch prompt for sanity check
        if i == 0 and rank == 0:
            print('\n' + '=' * 60)
            print('[DEBUG] First sample prompt (rank 0):')
            for msg in struct:
                print(f"  type={msg.get('type','?')}  value={str(msg.get('value',''))[:200]}")
            print('=' * 60 + '\n', flush=True)

        # If `SKIP_ERR` flag is set, bad video samples are dropped instead of aborting the whole run.
        try:
            response = model.generate(message=struct, dataset=dataset_name)
        except Exception as err:
            if not skip_err:
                raise
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            logging.warning(
                f'Skip sample {idx} in {model_name}/{dataset_name} during generation: '
                f'{type(err).__name__}: {err}'
            )
            response = ''
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in sample_indices_sub}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset,
        result_file_name,
        verbose=False,
        api_nproc=4,
        use_vllm=False):

    dataset_name = dataset.dataset_name
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, result_file_name)
    # Dump Predictions to Prev File if result file exists
    if osp.exists(result_file):
        return model

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{osp.splitext(result_file_name)[0]}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model,
        model_name=model_name,
        work_dir=work_dir,
        dataset=dataset,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc,
        use_vllm=use_vllm)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            for x in meta['index']:
                assert x in data_all
            meta['prediction'] = [str(data_all[x]) for x in meta['index']]
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model

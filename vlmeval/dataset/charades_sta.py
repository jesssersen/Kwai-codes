import re
import os.path as osp
from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

PRE_PROMPT = (
    'Please find the visual event described by a sentence in the video, '
    'determining its starting and ending times. '
    "The format should be: 'The event happens in the start time - end time seconds'. "
    "For example: The event 'person turn a light on' happens in the 24.3 - 30.4 seconds. "
    'Now I will give you the textual sentence: '
)

POST_PROMPT = 'Please return its start time and end time in seconds.'


def _compute_iou(pred_start, pred_end, gt_start, gt_end):
    """Compute temporal IoU between predicted and GT intervals."""
    intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0.0


def _parse_timestamps(text):
    """Extract (start, end) floats from model output.  Returns (None, None) on failure."""
    for pat in [
        r'(\d+\.?\d*)\s*[-\u2013]\s*(\d+\.?\d*)',   # "24.3 - 30.4" or "24.3–30.4"
        r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)',           # "24.3 to 30.4"
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            s, e = float(m.group(1)), float(m.group(2))
            if e >= s:
                return s, e
    return None, None


class CharadesSTA(VideoBaseDataset):
    """Charades-STA Temporal Grounding Benchmark.

    Given a video clip and a natural-language description, the model predicts
    the start and end timestamps (in seconds) of the described event.

    Dataset: lmms-lab/charades_sta on HuggingFace.
    Parquet fields: video (filename e.g. '0A8TF.mp4'), caption (str),
                    timestamp (list [start, end]).
    Videos: stored inside a Charades_v1_480/ subdirectory.

    Evaluation metrics: mIoU, R@1 IoU=0.3/0.5/0.7
    """

    TYPE = 'Video-VQA'

    def __init__(self, dataset='CharadesSTA', nframe=32, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['CharadesSTA']

    # ------------------------------------------------------------------
    #  Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self, dataset_name='CharadesSTA', repo_id='lmms-lab/charades_sta'):
        import glob
        import zipfile

        def find_parquets(root, split='test'):
            for pat in [
                osp.join(root, f'**/{split}-*.parquet'),
                osp.join(root, f'{split}-*.parquet'),
                osp.join(root, f'data/{split}-*.parquet'),
            ]:
                files = sorted(glob.glob(pat, recursive=True))
                if files:
                    return files
            return []

        def ensure_videos(root):
            video_dir = osp.join(root, 'Charades_v1_480')
            if osp.exists(video_dir) and len(os.listdir(video_dir)) > 0:
                return video_dir
            for zf in sorted(
                glob.glob(osp.join(root, '**/*.zip'), recursive=True)
                + glob.glob(osp.join(root, '*.zip'))
            ):
                bname = osp.basename(zf).lower()
                if 'charades' in bname or 'video' in bname:
                    print(f'CharadesSTA: extracting videos from {zf} ...')
                    with zipfile.ZipFile(zf, 'r') as z:
                        z.extractall(root)
                    if osp.exists(video_dir) and len(os.listdir(video_dir)) > 0:
                        return video_dir
            return video_dir  # may not exist; build_prompt will handle missing video gracefully

        def build_tsv(root, tsv_path):
            parquets = find_parquets(root, split='test')
            if not parquets:
                raise FileNotFoundError(
                    f'No test-split parquet annotation files found under {root}. '
                    f'Please verify that repo {repo_id} was downloaded successfully.'
                )
            df = pd.concat([pd.read_parquet(f) for f in parquets], ignore_index=True)
            rows = []
            for i, row in df.iterrows():
                vfile = str(row['video'])           # e.g. "0A8TF.mp4"
                vid_id = osp.splitext(vfile)[0]    # "0A8TF"
                ts = row['timestamp']              # list-like [start, end]
                rows.append({
                    'index': i,
                    'video': vid_id,
                    'video_file': vfile,
                    'question': str(row['caption']),
                    'answer': str(list(ts)),       # "[0.9, 6.0]"
                })
            pd.DataFrame(rows).to_csv(tsv_path, sep='\t', index=False)

        # 1. Locate or download from HuggingFace
        cache_path = get_cache_path(repo_id)
        if cache_path is None:
            print(f'Downloading {repo_id} from HuggingFace ...')
            cache_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

        # 2. Ensure video files are extracted if zipped
        video_dir = ensure_videos(cache_path)

        # 3. Build annotation TSV (idempotent)
        tsv_file = osp.join(cache_path, f'{dataset_name}.tsv')
        if not osp.exists(tsv_file):
            build_tsv(cache_path, tsv_file)

        return dict(data_file=tsv_file, root=video_dir)

    # ------------------------------------------------------------------
    #  Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_id = str(line['video'])
        video_path = osp.join(self.data_root, video_id + '.mp4')

        message = []
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            frames = self.save_video_frames(video_id)
            for frame in frames:
                message.append(dict(type='image', value=frame))

        text = f"{PRE_PROMPT}{line['question']}. {POST_PROMPT}"
        message.append(dict(type='text', value=text))
        return message

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            for idx, row in data.iterrows():
                pred = str(row.get('prediction', '')).strip()
                try:
                    gt_ts = eval(str(row.get('answer', '')))  # safe: stored as "[a, b]"
                    gt_start, gt_end = float(gt_ts[0]), float(gt_ts[1])
                except Exception:
                    data.loc[idx, 'iou'] = -1
                    continue
                p_start, p_end = _parse_timestamps(pred)
                if p_start is None:
                    data.loc[idx, 'iou'] = 0.0
                else:
                    data.loc[idx, 'iou'] = _compute_iou(p_start, p_end, gt_start, gt_end)
            dump(data, score_file)
        else:
            data = load(score_file)

        valid = data[data['iou'] >= 0]
        if len(valid) == 0:
            print('CharadesSTA: no valid predictions found.')
            return {}

        ious = valid['iou'].values
        result = {
            'mIoU': round(float(ious.mean()) * 100, 2),
            'R@1_IoU=0.3': round(float((ious >= 0.3).mean()) * 100, 2),
            'R@1_IoU=0.5': round(float((ious >= 0.5).mean()) * 100, 2),
            'R@1_IoU=0.7': round(float((ious >= 0.7).mean()) * 100, 2),
        }

        acc_file = get_intermediate_file_path(eval_file, '_acc')
        dump(pd.DataFrame([{'metric': k, 'value': v} for k, v in result.items()]), acc_file)

        print('CharadesSTA Evaluation Results:')
        for k, v in result.items():
            print(f'  {k}: {v:.2f}%')
        return result

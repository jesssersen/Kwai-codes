import re
import os.path as osp
from ..smp import *
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

SUBSETS = ['ReverseFilm', 'UCF101', 'Rtime_t2v', 'Rtime_v2t', 'AoTBench_QA']

# Maps dataset_name (as passed to __init__) → JSON filename stem
_DATASET_TO_FILE = {
    'AoTBench_ReverseFilm': 'ReverseFilm',
    'AoTBench_UCF101': 'UCF101',
    'AoTBench_Rtime_t2v': 'Rtime_t2v',
    'AoTBench_Rtime_v2t': 'Rtime_v2t',
    'AoTBench_QA': 'AoTBench_QA',
}


class AoTBench(VideoBaseDataset):
    """Arrow-of-Time Benchmark — temporal order understanding for video QA.

    Data layout expected under LMUDataRoot()/AoTBench/:
        data_files/{subset}.json   – annotations
        {video_name}               – video files (extensions may vary)

    JSON item fields: qa_idx (str/int), video_name (str), question (str), ans (str letter).
    """

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='AoTBench_ReverseFilm', nframe=16, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    # ------------------------------------------------------------------ #
    #  Registration
    # ------------------------------------------------------------------ #
    @classmethod
    def supported_datasets(cls):
        return list(_DATASET_TO_FILE.keys())

    # ------------------------------------------------------------------ #
    #  Data preparation: build TSV from source JSON
    # ------------------------------------------------------------------ #
    # Hardcoded server path; falls back to LMUDataRoot()/AoTBench if not present
    _DATA_ROOT = '/m2v_intern/xuboshen/zgw/data/AoTBench'

    def prepare_dataset(self, dataset_name='AoTBench_ReverseFilm'):
        if osp.exists(self._DATA_ROOT):
            data_root = self._DATA_ROOT
        else:
            lmu_root = LMUDataRoot()
            data_root = osp.join(lmu_root, 'AoTBench')

        assert dataset_name in _DATASET_TO_FILE, (
            f'Unknown AoTBench dataset: {dataset_name}. '
            f'Supported: {list(_DATASET_TO_FILE.keys())}'
        )
        subset_file = _DATASET_TO_FILE[dataset_name]

        json_file = osp.join(data_root, 'data_files', f'{subset_file}.json')
        tsv_file = osp.join(data_root, f'{dataset_name}.tsv')

        if not osp.exists(tsv_file):
            assert osp.exists(json_file), (
                f'AoTBench annotation not found: {json_file}\n'
                f'Please place AoTBench data under {data_root}/data_files/'
            )
            with open(json_file, 'r') as f:
                items = json.load(f)

            rows = []
            for item in items:
                video_name = item['video_name']
                # Strip extension for frame-cache directory naming
                video_id = osp.splitext(video_name)[0].replace('/', '_').replace('\\', '_')
                rows.append({
                    'index': item['qa_idx'],
                    'video': video_id,       # used by base-class frame cache
                    'video_name': video_name,  # original filename (with ext / subdir)
                    'question': item['question'],
                    'answer': item['ans'],
                })

            df = pd.DataFrame(rows)
            os.makedirs(data_root, exist_ok=True)
            df.to_csv(tsv_file, sep='\t', index=False)

        return dict(data_file=tsv_file, root=data_root)

    # ------------------------------------------------------------------ #
    #  Frame extraction (handles non-.mp4 extensions)
    # ------------------------------------------------------------------ #
    def save_video_frames(self, video):
        """Override to locate video with any extension under data_root."""
        import decord

        # Resolve video file: try original name stored in data, then common exts
        video_name = None
        matches = self.data[self.data['video'] == video]
        if len(matches) > 0 and 'video_name' in self.data.columns:
            video_name = matches.iloc[0]['video_name']

        if video_name and osp.exists(osp.join(self.data_root, video_name)):
            vid_path = osp.join(self.data_root, video_name)
        else:
            # Fallback: try appending common extensions
            for ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
                cand = osp.join(self.data_root, video + ext)
                if osp.exists(cand):
                    vid_path = cand
                    break
            else:
                raise FileNotFoundError(
                    f'Cannot find video "{video}" under {self.data_root}'
                )

        vid = decord.VideoReader(vid_path)

        if self.fps > 0:
            total_frames = len(vid)
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))
        else:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)

        flag = np.all([osp.exists(p) for p in frame_paths])
        if flag:
            return frame_paths

        import portalocker
        lock_path = osp.join(self.frame_root, video + '.lock')
        with portalocker.Lock(lock_path, 'w', timeout=30):
            if np.all([osp.exists(p) for p in frame_paths]):
                return frame_paths
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    # ------------------------------------------------------------------ #
    #  Prompt building
    # ------------------------------------------------------------------ #
    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_name = line.get('video_name', line['video'])
        if not osp.isabs(video_name):
            video_path = osp.join(self.data_root, video_name)
        else:
            video_path = video_name

        message = []
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            frames = self.save_video_frames(line['video'])
            for frame in frames:
                message.append(dict(type='image', value=frame))

        prompt = line['question'] + '\nAnswer with the option letter only.'
        message.append(dict(type='text', value=prompt))
        return message

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        correct = 0
        total = 0

        for _, row in data.iterrows():
            pred = str(row.get('prediction', '')).strip()
            ans = str(row.get('answer', '')).strip().upper()
            if not ans or not pred:
                continue
            m = re.search(r'[A-Za-z]', pred)
            pred_letter = m.group(0).upper() if m else ''
            correct += int(pred_letter == ans)
            total += 1

        acc = correct / total if total > 0 else 0.0
        result = {
            'accuracy': round(acc * 100, 2),
            'correct': correct,
            'total': total,
        }
        print(f'AoTBench Accuracy: {correct}/{total} = {acc:.2%}')
        return result

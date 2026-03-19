import re
import os.path as osp
from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

_OPTION_LETTERS = ['A', 'B', 'C']
_POST_PROMPT = '\nAnswer with the option\'s letter from the given choices directly.'


class PerceptionTest(VideoBaseDataset):
    """DeepMind PerceptionTest Video MCQ Benchmark (3-way multiple choice).

    Dataset: lmms-lab/PerceptionTest on HuggingFace.

    Supported splits:
        PerceptionTest_val  – ground-truth answers available; reports accuracy.
        PerceptionTest_test – no GT answers; saves a submission JSON file.

    Parquet fields (val): video_name, question, options (list[3]), question_id,
                          answer_id (int 0/1/2), area, reasoning, tag.
    Parquet fields (test): video_name, question, options, question_id.
    Videos: {dataset_root}/videos/{video_name}.mp4
    """

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='PerceptionTest_val', nframe=16, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['PerceptionTest_val', 'PerceptionTest_test']

    # ------------------------------------------------------------------
    #  Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self, dataset_name='PerceptionTest_val',
                        repo_id='lmms-lab/PerceptionTest'):
        import glob
        import zipfile

        split = 'val' if dataset_name == 'PerceptionTest_val' else 'test'

        def find_parquets(root, split_name):
            for pat in [
                osp.join(root, f'**/{split_name}-*.parquet'),
                osp.join(root, f'{split_name}-*.parquet'),
                osp.join(root, f'data/{split_name}-*.parquet'),
                osp.join(root, f'mc_question/{split_name}-*.parquet'),
            ]:
                files = sorted(glob.glob(pat, recursive=True))
                if files:
                    return files
            return []

        def ensure_videos(root):
            video_dir = osp.join(root, 'videos')
            if osp.exists(video_dir) and len(os.listdir(video_dir)) > 0:
                return video_dir
            for zf in sorted(
                glob.glob(osp.join(root, '**/*.zip'), recursive=True)
                + glob.glob(osp.join(root, '*.zip'))
            ):
                if 'video' in osp.basename(zf).lower():
                    print(f'PerceptionTest: extracting videos from {zf} ...')
                    with zipfile.ZipFile(zf, 'r') as z:
                        z.extractall(root)
                    if osp.exists(video_dir) and len(os.listdir(video_dir)) > 0:
                        return video_dir
            return video_dir  # may not exist yet; build_prompt handles missing file gracefully

        def build_tsv(root, tsv_path, split_name):
            parquets = find_parquets(root, split_name)
            if not parquets:
                raise FileNotFoundError(
                    f'No {split_name}-split parquet files found under {root}. '
                    f'Please verify that repo {repo_id} was downloaded successfully.'
                )
            df = pd.concat([pd.read_parquet(f) for f in parquets], ignore_index=True)
            has_answer = 'answer_id' in df.columns

            rows = []
            for i, row in df.iterrows():
                options = list(row['options'])
                # Build formatted question with labelled options
                q = str(row['question'])
                for letter, opt in zip(_OPTION_LETTERS, options[:3]):
                    q += f'\n{letter}. {opt}'
                q += _POST_PROMPT

                answer_id = int(row['answer_id']) if has_answer else -1
                answer = _OPTION_LETTERS[answer_id] if 0 <= answer_id < 3 else ''

                rec = {
                    'index': i,
                    'video': str(row['video_name']),
                    'question': q,
                    'answer': answer,
                    'question_id': str(row['question_id']),
                    'answer_id': answer_id,
                }
                for field in ('area', 'reasoning', 'tag'):
                    rec[field] = str(row[field]) if field in df.columns else ''
                rows.append(rec)

            pd.DataFrame(rows).to_csv(tsv_path, sep='\t', index=False)

        # 1. Locate or download from HuggingFace
        cache_path = get_cache_path(repo_id)
        if cache_path is None:
            print(f'Downloading {repo_id} from HuggingFace ...')
            cache_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

        # 2. Ensure video files are present / extracted
        video_dir = ensure_videos(cache_path)

        # 3. Build annotation TSV (idempotent)
        tsv_file = osp.join(cache_path, f'{dataset_name}.tsv')
        if not osp.exists(tsv_file):
            build_tsv(cache_path, tsv_file, split)

        return dict(data_file=tsv_file, root=video_dir)

    # ------------------------------------------------------------------
    #  Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_name = str(line['video'])
        video_path = osp.join(self.data_root, video_name + '.mp4')
        if not osp.exists(video_path):
            alt = osp.join(self.data_root, video_name + '.MP4')
            if osp.exists(alt):
                video_path = alt

        message = []
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            frames = self.save_video_frames(video_name)
            for frame in frames:
                message.append(dict(type='image', value=frame))

        message.append(dict(type='text', value=str(line['question'])))
        return message

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)

        # ---- Test split: save submission (no GT answers) ----
        if 'answer_id' not in data.columns or (data['answer_id'].astype(int) == -1).all():
            sub_file = get_intermediate_file_path(eval_file, '_submission', 'json')
            submission = []
            for _, row in data.iterrows():
                pred = str(row.get('prediction', '')).strip()
                m = re.search(r'[A-Ca-c]', pred)
                pred_letter = m.group(0).upper() if m else ''
                pred_id = _OPTION_LETTERS.index(pred_letter) if pred_letter in _OPTION_LETTERS else -1
                submission.append({
                    'question_id': str(row.get('question_id', '')),
                    'answer': pred_id,
                })
            with open(sub_file, 'w') as f:
                json.dump(submission, f, indent=2)
            print(f'PerceptionTest test submission saved to {sub_file}')
            return {}

        # ---- Val split: compute accuracy ----
        score_file = get_intermediate_file_path(eval_file, '_score')
        if not osp.exists(score_file):
            for idx, row in data.iterrows():
                pred = str(row.get('prediction', '')).strip()
                m = re.search(r'[A-Ca-c]', pred)
                pred_letter = m.group(0).upper() if m else ''
                pred_id = (
                    _OPTION_LETTERS.index(pred_letter)
                    if pred_letter in _OPTION_LETTERS else -1
                )
                data.loc[idx, 'pred_id'] = pred_id
                data.loc[idx, 'score'] = int(pred_id == int(row.get('answer_id', -1)))
            dump(data, score_file)
        else:
            data = load(score_file)

        def acc(sub):
            valid = sub[sub['answer_id'].astype(int) >= 0]
            return round(valid['score'].sum() / len(valid) * 100, 2) if len(valid) > 0 else 0.0

        result_rows = [{'split': 'Overall', 'category': 'all', 'accuracy': acc(data)}]
        for cat_col in ('area', 'reasoning', 'tag'):
            if cat_col in data.columns:
                for val in sorted(data[cat_col].dropna().unique()):
                    result_rows.append({
                        'split': cat_col,
                        'category': val,
                        'accuracy': acc(data[data[cat_col] == val]),
                    })

        result_df = pd.DataFrame(result_rows)
        acc_file = get_intermediate_file_path(eval_file, '_acc')
        dump(result_df, acc_file)

        overall = acc(data)
        print(f'PerceptionTest Val Accuracy: {overall:.2f}%')
        return result_df

import re
import os.path as osp
from ..smp import *
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

_PROMPT_VL = (
    'These are the frames of a video. '
    'Select the best answer to the following multiple-choice question based on the video. '
    'Respond with only the letter (A, B, C, D, E, F) of the correct option.'
)

_OPTION_LETTERS = list('ABCDEF')


class FutureOmni(VideoBaseDataset):
    """FutureOmni video-QA benchmark (up to 6-option MCQ, video-only path).

    Data layout expected under LMUDataRoot()/FutureOmni/:
        futureomni_test.json   – annotations
        videos/{qid}.mp4       – video clips

    JSON item fields:
        qid (int), source (str), question (str), options (list[str]),
        video (str, original name), seconds (float, clip end time), answer (str letter A-F).
    """

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='FutureOmni', nframe=32, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    # ------------------------------------------------------------------ #
    #  Registration
    # ------------------------------------------------------------------ #
    @classmethod
    def supported_datasets(cls):
        return ['FutureOmni']

    # ------------------------------------------------------------------ #
    #  Data preparation: build TSV from source JSON
    # ------------------------------------------------------------------ #
    # Hardcoded server path; falls back to LMUDataRoot()/FutureOmni if not present
    _DATA_ROOT = '/m2v_intern/xuboshen/zgw/data/FutureOmni'

    def prepare_dataset(self, dataset_name='FutureOmni'):
        if osp.exists(self._DATA_ROOT):
            data_root = self._DATA_ROOT
        else:
            lmu_root = LMUDataRoot()
            data_root = osp.join(lmu_root, 'FutureOmni')
        video_root = osp.join(data_root, 'videos')

        json_file = osp.join(data_root, 'futureomni_test.json')
        tsv_file = osp.join(data_root, f'{dataset_name}.tsv')

        if not osp.exists(tsv_file):
            assert osp.exists(json_file), (
                f'FutureOmni annotation not found: {json_file}\n'
                f'Please place futureomni_test.json under {data_root}/'
            )
            with open(json_file, 'r') as f:
                items = json.load(f)

            rows = []
            for idx, item in enumerate(items):
                qid = item['qid']
                options = item.get('options', [])
                # options already contain letter prefixes (e.g. "A. xxx"), join directly
                option_str = '\n'.join(options[:len(_OPTION_LETTERS)])
                question_full = item['question'].rstrip() + '\n' + option_str

                # GT answer: normalise to single uppercase letter
                raw_ans = item.get('answer', item.get('gt_answer', item.get('ans', '')))
                answer = str(raw_ans).strip().upper()
                if answer and answer[0] in _OPTION_LETTERS:
                    answer = answer[0]

                rows.append({
                    'index': idx,
                    'video': str(qid),       # e.g. "0", "1000"
                    'question': question_full,
                    'answer': answer,
                    'seconds': item.get('seconds', None),
                    'source': item.get('source', ''),
                    'qid': qid,
                })

            df = pd.DataFrame(rows)
            os.makedirs(data_root, exist_ok=True)
            df.to_csv(tsv_file, sep='\t', index=False)

        return dict(data_file=tsv_file, root=video_root)

    # ------------------------------------------------------------------ #
    #  Prompt building
    # ------------------------------------------------------------------ #
    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]

        qid = line.get('qid', line['video'])
        video_path = osp.join(self.data_root, f'{qid}.mp4')

        message = [dict(type='text', value=_PROMPT_VL)]

        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            frames = self.save_video_frames(line['video'])
            for frame in frames:
                message.append(dict(type='image', value=frame))

        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        return message

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)

        # Per-source breakdown
        source_stats: dict = {}
        correct = 0
        total = 0

        for _, row in data.iterrows():
            pred = str(row.get('prediction', '')).strip()
            ans = str(row.get('answer', '')).strip().upper()
            if not ans:
                continue

            m = re.search(r'[A-Fa-f]', pred)
            pred_letter = m.group(0).upper() if m else ''
            hit = int(pred_letter == ans)

            src = str(row.get('source', 'all'))
            if src not in source_stats:
                source_stats[src] = {'correct': 0, 'total': 0}
            source_stats[src]['correct'] += hit
            source_stats[src]['total'] += 1

            correct += hit
            total += 1

        acc = correct / total if total > 0 else 0.0
        result = {'accuracy': round(acc * 100, 2), 'correct': correct, 'total': total}

        for src, stats in sorted(source_stats.items()):
            src_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            result[f'accuracy_{src}'] = round(src_acc * 100, 2)

        print(f'FutureOmni Accuracy: {correct}/{total} = {acc:.2%}')
        for src, stats in sorted(source_stats.items()):
            print(f'  [{src}] {stats["correct"]}/{stats["total"]}')

        return result

from mmengine.config import read_base
from opencompass.datasets import (
    GSM8KDataset,
    IFEvalDataset,
    IFEvaluator,
    MATHEvaluator,
    gsm8k_dataset_postprocess,
    math_postprocess_sdar,
    math_postprocess_v2,
)
from opencompass.models import LMDeploywithChatTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

_os = __import__('os')
_pathlib = __import__('pathlib')


def _default_num_workers() -> int:
    visible_devices = _os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible_devices:
        return max(1, len([x for x in visible_devices.split(',') if x.strip()]))
    return 1


def _format_threshold(threshold: float) -> str:
    return f'{threshold:.2f}'.replace('.', '_')


REPO_ROOT = _pathlib.Path(__file__).resolve().parents[3]
REPO_ROOT = str(REPO_ROOT)
MODEL_ROOT = _os.path.abspath(
    _os.environ.get('SDAR_MODEL_ROOT', _os.path.join(REPO_ROOT, 'Models'))
)
MODEL_NAME = _os.environ.get('SDAR_MODEL_NAME', 'SDAR-1.7B-Chat')
MODEL_PATH = _os.path.abspath(_os.path.join(MODEL_ROOT, MODEL_NAME))
EVAL_SCOPE = _os.environ.get('SDAR_EVAL_SCOPE', 'gsm8k').lower()
NUM_WORKERS = int(_os.environ.get('SDAR_EVAL_GPUS', _default_num_workers()))
INFER_BATCH_SIZE = int(_os.environ.get('SDAR_INFER_BATCH_SIZE', '1'))
TP = int(_os.environ.get('SDAR_LMDEPLOY_TP', '1'))
BLOCK_LENGTH = int(_os.environ.get('SDAR_BLOCK_LENGTH', '4'))
CONFIDENCE_THRESHOLD = float(_os.environ.get('SDAR_CONFIDENCE_THRESHOLD', '0.95'))
MAX_NEW_TOKENS = int(_os.environ.get('SDAR_MAX_NEW_TOKENS', '4096'))
TOP_P = float(_os.environ.get('SDAR_TOP_P', '0.95'))
TOP_K = int(_os.environ.get('SDAR_TOP_K', '50'))
TEMPERATURE = float(_os.environ.get('SDAR_TEMPERATURE', '1.0'))

if not _os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f'SDAR model path does not exist: {MODEL_PATH}. '
        'Set SDAR_MODEL_ROOT or SDAR_MODEL_NAME before running OpenCompass.'
    )

if not (0 < CONFIDENCE_THRESHOLD <= 1.0):
    raise ValueError('SDAR_CONFIDENCE_THRESHOLD must be in the range (0, 1].')

if EVAL_SCOPE not in {'gsm8k', 'full'}:
    raise ValueError("SDAR_EVAL_SCOPE must be either 'gsm8k' or 'full'.")

if EVAL_SCOPE == 'full':
    with read_base():
        from opencompass.configs.datasets.MathBench.mathbench_2024_gen_50a320 import (
            mathbench_datasets,
        )
        from opencompass.configs.datasets.gsm8k.gsm8k_0shot_v2_gen_17d799 import (
            gsm8k_datasets,
        )
        from opencompass.configs.datasets.humaneval.humaneval_gen import (
            humaneval_datasets,
        )
        from opencompass.configs.datasets.math.math_prm800k_500_0shot_cot_gen_11c4b5 import (
            math_datasets,
        )
        from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_0shot_nocot_gen_a2e416 import (
            sanitized_mbpp_datasets,
        )
        from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import (
            mmlu_datasets,
        )
        from opencompass.configs.summarizers.groups.mathbench_v1_2024 import (
            mathbench_2024_summary_groups,
        )
        from opencompass.configs.summarizers.groups.mmlu import (
            mmlu_summary_groups,
        )
else:
    gsm8k_datasets = [
        dict(
            abbr='gsm8k',
            type=GSM8KDataset,
            path='opencompass/gsm8k',
            reader_cfg=dict(input_columns=['question'], output_column='answer'),
            infer_cfg=dict(
                prompt_template=dict(
                    type=PromptTemplate,
                    template=dict(
                        round=[
                            dict(
                                role='HUMAN',
                                prompt=(
                                    '{question}\nSolve the problem step by step, '
                                    'but keep the reasoning concise. Once you '
                                    'know the answer, put the final answer '
                                    'within \\boxed{} and stop immediately. '
                                    'Do not provide alternative '
                                    'interpretations, repeated checks, or any '
                                    'text after the boxed answer.'
                                ),
                            ),
                        ],
                    ),
                ),
                retriever=dict(type=ZeroRetriever),
                inferencer=dict(type=GenInferencer, max_out_len=MAX_NEW_TOKENS),
            ),
            eval_cfg=dict(
                evaluator=dict(type=MATHEvaluator, version='v2'),
                pred_postprocessor=dict(type=math_postprocess_sdar),
                dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
            ),
        )
    ]


ifeval_datasets = [
    dict(
        abbr='IFEval',
        type=IFEvalDataset,
        path='data/ifeval/input_data.jsonl',
        reader_cfg=dict(input_columns=['prompt'], output_column='reference'),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(round=[dict(role='HUMAN', prompt='{prompt}')]),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=MAX_NEW_TOKENS),
        ),
        eval_cfg=dict(
            evaluator=dict(type=IFEvaluator),
            pred_role='BOT',
        ),
    )
]


full_summary_groups = sum(
    [v for k, v in locals().items() if k.endswith('_summary_groups')], []
)
full_summary_groups.append(
    {
        'name': 'Mathbench',
        'subsets': ['mathbench-a (average)', 'mathbench-t (average)'],
    },
)

full_summarizer = dict(
    dataset_abbrs=[
        'Instruction Following',
        ['IFEval', 'Prompt-level-strict-accuracy'],
        '',
        'Math Calculation',
        ['gsm8k', 'accuracy'],
        ['Mathbench', 'naive_average'],
        ['math_prm800k_500', 'accuracy'],
        '',
        'Knowledge',
        ['mmlu', 'naive_average'],
        '',
        'Code',
        ['openai_humaneval', 'humaneval_pass@1'],
        ['sanitized_mbpp', 'score'],
        '',
        'mmlu',
        'mmlu-stem',
        'mmlu-social-science',
        'mmlu-humanities',
        'mmlu-other',
        '',
        '###### MathBench-A: Application Part ######',
        'college',
        'high',
        'middle',
        'primary',
        'arithmetic',
        'mathbench-a (average)',
        '###### MathBench-T: Theory Part ######',
        'college_knowledge',
        'high_knowledge',
        'middle_knowledge',
        'primary_knowledge',
        'mathbench-t (average)',
    ],
    summary_groups=full_summary_groups,
)

gsm8k_summarizer = dict(
    dataset_abbrs=[['gsm8k', 'accuracy']],
    summary_groups=[],
)

if EVAL_SCOPE == 'full':
    datasets = [
        *mmlu_datasets,
        *gsm8k_datasets,
        *humaneval_datasets,
        *sanitized_mbpp_datasets,
        *math_datasets,
        *mathbench_datasets,
        *ifeval_datasets,
    ]
    summarizer = full_summarizer
else:
    datasets = [*gsm8k_datasets]
    summarizer = gsm8k_summarizer

for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = INFER_BATCH_SIZE

dllm_unmasking_strategy = (
    'low_confidence_dynamic'
    if CONFIDENCE_THRESHOLD < 1.0
    else 'low_confidence_static'
)
model_abbr = (
    f'{MODEL_NAME}-b{BLOCK_LENGTH}-thr{_format_threshold(CONFIDENCE_THRESHOLD)}'
)

models = [
    dict(
        type=LMDeploywithChatTemplate,
        abbr=model_abbr,
        path=MODEL_PATH,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(
            top_p=TOP_P,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        ),
        model_kwargs=dict(
            tp=TP,
            dtype='float16',
            dllm_block_length=BLOCK_LENGTH,
            dllm_denoising_steps=BLOCK_LENGTH,
            dllm_confidence_threshold=CONFIDENCE_THRESHOLD,
            dllm_unmasking_strategy=dllm_unmasking_strategy,
        ),
    )
]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=NUM_WORKERS,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=NUM_WORKERS,
        keep_tmp_file=True,
        task=dict(type=OpenICLInferTask),
        retry=5,
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=1),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)

work_dir = _os.environ.get('SDAR_WORK_DIR', './outputs/eval-chat-sdar')

del _default_num_workers
del _format_threshold
del _os
del _pathlib

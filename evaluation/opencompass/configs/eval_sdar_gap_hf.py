from opencompass.datasets import (
    GSM8KDataset,
    MATHEvaluator,
    gsm8k_dataset_postprocess,
    math_postprocess_v2,
)
from opencompass.models import SDARGapwithChatTemplate
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


REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[3])
MODEL_ROOT = _os.path.abspath(
    _os.environ.get('SDAR_MODEL_ROOT', _os.path.join(REPO_ROOT, 'Models'))
)
MODEL_NAME = _os.environ.get('SDAR_MODEL_NAME', 'SDAR-1.7B-Chat')
MODEL_PATH = _os.path.abspath(
    _os.environ.get('SDAR_MODEL_PATH', _os.path.join(MODEL_ROOT, MODEL_NAME))
)
NUM_WORKERS = int(_os.environ.get('SDAR_EVAL_GPUS', _default_num_workers()))
INFER_BATCH_SIZE = int(_os.environ.get('SDAR_INFER_BATCH_SIZE', '1'))
BLOCK_LENGTH = int(_os.environ.get('SDAR_BLOCK_LENGTH', '4'))
CONFIDENCE_THRESHOLD = float(_os.environ.get('SDAR_CONFIDENCE_THRESHOLD', '0.95'))
REMASK_THRESHOLD = float(_os.environ.get('SDAR_REMASK_THRESHOLD', '0.5'))
MAX_NEW_TOKENS = int(_os.environ.get('SDAR_MAX_NEW_TOKENS', '1024'))
TOP_P = float(_os.environ.get('SDAR_TOP_P', '1.0'))
TOP_K = int(_os.environ.get('SDAR_TOP_K', '1'))
TEMPERATURE = float(_os.environ.get('SDAR_TEMPERATURE', '1.0'))
TORCH_DTYPE = _os.environ.get('SDAR_TORCH_DTYPE', 'bfloat16')
TEST_RANGE = _os.environ.get('SDAR_TEST_RANGE')

if not _os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f'SDAR model path does not exist: {MODEL_PATH}. '
        'Set SDAR_MODEL_ROOT or SDAR_MODEL_NAME before running OpenCompass.'
    )

if INFER_BATCH_SIZE < 1:
    raise ValueError('SDAR_INFER_BATCH_SIZE must be a positive integer.')

torch_dtype = {
    'float16': 'torch.float16',
    'bfloat16': 'torch.bfloat16',
    'float32': 'torch.float',
}.get(TORCH_DTYPE)
if torch_dtype is None:
    raise ValueError("SDAR_TORCH_DTYPE must be one of: float16, bfloat16, float32.")

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
                                '{question}\nPlease reason step by step, '
                                'and put your final answer within '
                                '\\boxed{}.'
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
            pred_postprocessor=dict(type=math_postprocess_v2),
            dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
        ),
    )
]

datasets = [*gsm8k_datasets]
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = INFER_BATCH_SIZE
    if TEST_RANGE:
        dataset['reader_cfg']['test_range'] = TEST_RANGE

model_abbr = (
    f'{MODEL_NAME}-gap-b{BLOCK_LENGTH}-thr{_format_threshold(CONFIDENCE_THRESHOLD)}'
    f'-rt{_format_threshold(REMASK_THRESHOLD)}'
)

models = [
    dict(
        type=SDARGapwithChatTemplate,
        abbr=model_abbr,
        path=MODEL_PATH,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(
            mask_id=151669,
            gen_length=MAX_NEW_TOKENS,
            block_length=BLOCK_LENGTH,
            denoising_steps=BLOCK_LENGTH,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            remasking_strategy='low_confidence_dynamic'
            if CONFIDENCE_THRESHOLD < 1.0
            else 'low_confidence_static',
            confidence_threshold=CONFIDENCE_THRESHOLD,
            remask_threshold=REMASK_THRESHOLD,
        ),
        model_kwargs=dict(
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map='cuda',
        ),
    )
]

summarizer = dict(
    dataset_abbrs=[['gsm8k', 'accuracy']],
    summary_groups=[],
)

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

work_dir = _os.environ.get('SDAR_WORK_DIR', './outputs/eval-chat-sdar-gap')

del _default_num_workers
del _format_threshold
del _os
del _pathlib

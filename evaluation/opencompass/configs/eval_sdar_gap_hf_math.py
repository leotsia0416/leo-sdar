from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_sdar
from opencompass.models import SDARGapwithChatTemplate
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

_os = __import__('os')
_pathlib = __import__('pathlib')
_json = __import__('json')


def _default_num_workers() -> int:
    visible_devices = _os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible_devices:
        return max(1, len([x for x in visible_devices.split(',') if x.strip()]))
    return 1


def _format_threshold(threshold: float) -> str:
    return f'{threshold:.2f}'.replace('.', '_')


def _has_model_weights(model_path: str) -> bool:
    if _os.path.exists(_os.path.join(model_path, 'model.safetensors')):
        return True
    if _os.path.exists(_os.path.join(model_path, 'model.safetensors.index.json')):
        return True
    if _os.path.exists(_os.path.join(model_path, 'pytorch_model.bin')):
        return True
    if _os.path.exists(_os.path.join(model_path, 'pytorch_model.bin.index.json')):
        return True
    return any(_pathlib.Path(model_path).glob('model-*.safetensors'))


def _load_model_config(model_path: str) -> dict:
    with open(_os.path.join(model_path, 'config.json'), 'r', encoding='utf-8') as f:
        return _json.load(f)


def _config_default(model_config: dict, key: str, fallback):
    value = model_config.get(key)
    return fallback if value is None else value


REPO_ROOT = str(_pathlib.Path(__file__).resolve().parents[3])
MODEL_ROOT = _os.path.abspath(
    _os.environ.get('SDAR_MODEL_ROOT', _os.path.join(REPO_ROOT, 'Models'))
)
MODEL_NAME = _os.environ.get('SDAR_MODEL_NAME', 'SDAR-1.7B-Chat-')
MODEL_PATH = _os.path.abspath(
    _os.environ.get('SDAR_MODEL_PATH', _os.path.join(MODEL_ROOT, MODEL_NAME))
)
MATH_PATH = _os.environ.get('SDAR_MATH_PATH', 'opencompass/math')
MATH_ABBR = _os.environ.get('SDAR_MATH_ABBR', 'math')

if not _os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f'SDAR model path does not exist: {MODEL_PATH}. '
        'Set SDAR_MODEL_ROOT or SDAR_MODEL_NAME before running OpenCompass.'
    )
if not _os.path.exists(_os.path.join(MODEL_PATH, 'config.json')):
    raise FileNotFoundError(f'SDAR model path is missing config.json: {MODEL_PATH}.')
if not _has_model_weights(MODEL_PATH):
    raise FileNotFoundError(f'SDAR model path is missing model weights: {MODEL_PATH}.')

MODEL_CONFIG = _load_model_config(MODEL_PATH)
NUM_WORKERS = int(_os.environ.get('SDAR_EVAL_GPUS', _default_num_workers()))
INFER_BATCH_SIZE = int(_os.environ.get('SDAR_INFER_BATCH_SIZE', '1'))
BLOCK_LENGTH = int(
    _os.environ.get('SDAR_BLOCK_LENGTH', str(_config_default(MODEL_CONFIG, 'block_size', 4)))
)
CONFIDENCE_THRESHOLD = float(
    _os.environ.get(
        'SDAR_CONFIDENCE_THRESHOLD',
        str(_config_default(MODEL_CONFIG, 'gap_rollout_confidence_threshold', 0.95)),
    )
)
REMASK_THRESHOLD = float(
    _os.environ.get(
        'SDAR_REMASK_THRESHOLD',
        str(_config_default(MODEL_CONFIG, 'gap_remask_threshold', 0.5)),
    )
)
REMASK_START_RATIO = float(_os.environ.get('SDAR_REMASK_START_RATIO', '0.0'))
REMASK_INTERVAL_BLOCKS = int(_os.environ.get('SDAR_REMASK_INTERVAL_BLOCKS', '1'))
REMASK_WINDOW_BLOCKS = int(
    _os.environ.get(
        'SDAR_REMASK_WINDOW_BLOCKS',
        str(_config_default(MODEL_CONFIG, 'gap_remask_window_blocks', 5)),
    )
)
REMASK_START_TOKENS = int(_os.environ.get('SDAR_REMASK_START_TOKENS', '192'))
REMASK_PREFIX_GUARD_TOKENS = int(
    _os.environ.get('SDAR_REMASK_PREFIX_GUARD_TOKENS', str(REMASK_START_TOKENS))
)
REMASK_TAIL_GUARD_BLOCKS = int(_os.environ.get('SDAR_REMASK_TAIL_GUARD_BLOCKS', '1'))
ROLLOUT_STRATEGY = _os.environ.get(
    'SDAR_ROLLOUT_STRATEGY',
    str(_config_default(MODEL_CONFIG, 'gap_rollout_strategy', 'low_confidence_dynamic')),
)
MAX_NEW_TOKENS = int(_os.environ.get('SDAR_MAX_NEW_TOKENS', '1536'))
TOP_P = float(_os.environ.get('SDAR_TOP_P', '1.0'))
TOP_K = int(_os.environ.get('SDAR_TOP_K', '1'))
TEMPERATURE = float(_os.environ.get('SDAR_TEMPERATURE', '0.0'))
TORCH_DTYPE = _os.environ.get('SDAR_TORCH_DTYPE', 'bfloat16')
TEST_RANGE = _os.environ.get('SDAR_TEST_RANGE')

if INFER_BATCH_SIZE < 1:
    raise ValueError('SDAR_INFER_BATCH_SIZE must be a positive integer.')
if not 0.0 <= REMASK_START_RATIO <= 1.0:
    raise ValueError('SDAR_REMASK_START_RATIO must be within [0.0, 1.0].')
if REMASK_START_TOKENS < 0:
    raise ValueError('SDAR_REMASK_START_TOKENS must be non-negative.')
if REMASK_PREFIX_GUARD_TOKENS < 0:
    raise ValueError('SDAR_REMASK_PREFIX_GUARD_TOKENS must be non-negative.')
if REMASK_TAIL_GUARD_BLOCKS < 0:
    raise ValueError('SDAR_REMASK_TAIL_GUARD_BLOCKS must be non-negative.')

torch_dtype = {
    'float16': 'torch.float16',
    'bfloat16': 'torch.bfloat16',
    'float32': 'torch.float',
}.get(TORCH_DTYPE)
if torch_dtype is None:
    raise ValueError("SDAR_TORCH_DTYPE must be one of: float16, bfloat16, float32.")

math_datasets = [
    dict(
        abbr=MATH_ABBR,
        type=MATHDataset,
        path=MATH_PATH,
        reader_cfg=dict(input_columns=['problem'], output_column='solution'),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(
                            role='HUMAN',
                            prompt=(
                                '{problem}\nSolve the problem step by step, '
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
        ),
    )
]

datasets = [*math_datasets]
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = INFER_BATCH_SIZE
    if TEST_RANGE:
        dataset['reader_cfg']['test_range'] = TEST_RANGE

model_abbr = (
    f'{MODEL_NAME}-gap-b{BLOCK_LENGTH}-thr{_format_threshold(CONFIDENCE_THRESHOLD)}'
    f'-rt{_format_threshold(REMASK_THRESHOLD)}'
    f'-t{_format_threshold(TEMPERATURE)}'
    f'-rs{_format_threshold(REMASK_START_RATIO)}'
    f'-ri{REMASK_INTERVAL_BLOCKS}'
    f'-rw{REMASK_WINDOW_BLOCKS}'
    f'-rstk{REMASK_START_TOKENS}'
    f'-pg{REMASK_PREFIX_GUARD_TOKENS}'
    f'-tg{REMASK_TAIL_GUARD_BLOCKS}'
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
            remasking_strategy=ROLLOUT_STRATEGY,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            remask_threshold=REMASK_THRESHOLD,
            remask_start_ratio=REMASK_START_RATIO,
            remask_interval_blocks=REMASK_INTERVAL_BLOCKS,
            remask_window_blocks=REMASK_WINDOW_BLOCKS,
            remask_start_tokens=REMASK_START_TOKENS,
            remask_prefix_guard_tokens=REMASK_PREFIX_GUARD_TOKENS,
            remask_tail_guard_blocks=REMASK_TAIL_GUARD_BLOCKS,
        ),
        model_kwargs=dict(
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map='cuda',
        ),
    )
]

summarizer = dict(
    dataset_abbrs=[[MATH_ABBR, 'accuracy']],
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

work_dir = _os.environ.get('SDAR_WORK_DIR', './outputs/math_test_gap')

globals().pop('_default_num_workers', None)
globals().pop('_format_threshold', None)
globals().pop('_has_model_weights', None)
globals().pop('_load_model_config', None)
globals().pop('_config_default', None)
globals().pop('_os', None)
globals().pop('_pathlib', None)
globals().pop('_json', None)

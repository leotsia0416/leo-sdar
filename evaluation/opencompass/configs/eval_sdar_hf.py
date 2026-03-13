import torch
from opencompass.datasets import (
    GSM8KDataset,
    IFEvalDataset,
    IFEvaluator,
    MATHEvaluator,
    gsm8k_dataset_postprocess,
    math_postprocess_v2,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import BD3withChatTemplate


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
                                '{question}\nPlease reason step by step, and '
                                'put your final answer within \\boxed{}.'
                            ),
                        ),
                    ],
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=MATHEvaluator, version='v2'),
            pred_postprocessor=dict(type=math_postprocess_v2),
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
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=IFEvaluator),
            pred_role='BOT',
        ),
    )
]

# Summarizer
summarizer = dict(
    dataset_abbrs=[['gsm8k', 'accuracy']],
    summary_groups=[],
)

# datasets = [*mmlu_datasets, *gsm8k_datasets, *humaneval_datasets, *sanitized_mbpp_datasets, *math_datasets, *mathbench_datasets, *ifeval_datasets]
datasets = [*gsm8k_datasets]
for dataset in datasets:
    dataset['infer_cfg']['inferencer']['batch_size'] = 1 # only support batchsize=1 up to now

# model
model_configs = [
    # ("SDAR-1.7B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-1.7B-Chat", 4, 0.95, 1),
    ("SDAR-1.7B-Chat-b4-thr1_00", "/work/tom900908/SDAR/Models/SDAR-1.7B-Chat", 4, 1.0, 1),
    # ("SDAR-4B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-4B-Chat", 4, 0.95, 1),
    # ("SDAR-4B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-4B-Chat", 4, 1.0, 1),
    # ("SDAR-8B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-8B-Chat", 4, 0.95, 1),
    # ("SDAR-8B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-8B-Chat", 4, 1.0, 1),
    # ("SDAR-30B-A3B-Chat-b4-thr0_95", "/xxx/Models/SDAR/SDAR-30B-A3B-Chat", 4, 0.95, 1),
    # ("SDAR-30B-A3B-Chat-b4-thr1_00", "/xxx/Models/SDAR/SDAR-30B-A3B-Chat", 4, 1.0, 1)
]
models = []
for abbr, path, block_length, threshold, num_gpus in model_configs:

    models.append(
        dict(
        type=BD3withChatTemplate,
        abbr=abbr,
        path=path,
        run_cfg=dict(num_gpus=num_gpus),
        generation_kwargs=dict(
            mask_id=151669,
            gen_length=4096,
            block_length=block_length,
            denoising_steps=block_length,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            cfg_scale=0.0,
            remasking='low_confidence',
            threshold=threshold
        ),
        model_kwargs=dict(
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ),
    )
    )

GPUS = 2
infer = dict(
    # 同时启动num_workers个任务并行
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=GPUS,  # 划分完成后的任务数 / 预期能有的 worker 数
        # force_rebuild=True
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=GPUS,  # 最大并行运行进程数
        keep_tmp_file=True,
        task=dict(type=OpenICLInferTask),
        retry=5
    )
)
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=16),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask, dump_details=True)),
)

work_dir = f'./outputs/eval-chat-sdar'

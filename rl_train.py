import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['FONTCONFIG_PATH'] = '/etc/fonts'
os.environ['FONTCONFIG_FILE'] = '/etc/fonts/fonts.conf'
import torch
import torch.optim as optim
from dataclasses import dataclass

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, get_constant_schedule

from trl import TrlParser
from trl.trainer.grpo_config import GRPOConfig
from datasets import load_from_disk

from rewards import get_reward_function
from grpo_trainer import TopSampleGRPOTrainer
from metrics_async import init_pool, close_pool

SEED = 16
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" # processor used

@dataclass
class RewardArgs:
    failure_reward: float = -10.0
    iou_coef: float = 10.0
    cd_coef: float = 0.0
    auc_coef: float = 0.0
    aoc_gms_coef: float = 0.0
    mae_coef: float = 0.0
    get_nc: bool = False
    nc_n_points: int = 16384
    nc_tol: int = 5
    print_sample_steps: int = 25
    pool_size: int = 16
    r_mode: str = "10_iou"
    get_aoc_gms: bool = False
    aoc_gms_n_points: int = 8192
    aoc_gms_n_angles: int = 125
    aoc_gms_rel_tol: float = 0.05
    aoc_gms_cube_trick: bool = True
    aoc_gms_pc_cache_enable: bool = False
    aoc_gms_upper_bound_tol_rt: int = 25
    aoc_gms_autofix_sampling: bool = False
    get_mae_render: bool = False

@dataclass
class TrainingArgs:
    sft_path: str = ""
    clip_cov: bool = False
    top_samples: int = 4
    resume_ckpt_path: str = ""

parser = TrlParser((GRPOConfig, RewardArgs, TrainingArgs))
grpo, rargs, targs = parser.parse_args_and_config()

init_pool(rargs.pool_size)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True,
                                            #min_pixels=256*28*28,
                                            #max_pixels=1280*28*28,
                                            resized_width=14 * 17 * 2,
                                            resized_height=14 * 17 * 4,
                                            padding_side="left",)


model = Qwen2VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=targs.sft_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True, 
)
model.enable_input_require_grads() 
model.gradient_checkpointing_enable()

HF_DATASET = "/scratch/498rustam/datasets/rendered_cadevolve_normalized_1_1_fixed" # ME: change this

if "dp_f360" in grpo.output_dir :
    print("Training on only DeepCad and F360, no MCB")
    HF_DATASET = "/scratch/498rustam/datasets/rendered_cadevolve_normalized_1_1_deepcadf360" # ME: change this TODO: fix dataset for msu

hf_dataset = load_from_disk(HF_DATASET)

optimizer = optim.AdamW(model.parameters(), lr=grpo.learning_rate)
lr_scheduler = get_constant_schedule(optimizer)

nc_params = {
    "get_nc": rargs.get_nc,
    "n_points": rargs.nc_n_points,
    "tol": rargs.nc_tol,
    "get_mae_render": rargs.get_mae_render,
    "get_aoc_gms": rargs.get_aoc_gms,
    "aoc_gms_n_points": rargs.aoc_gms_n_points,
    "aoc_gms_rel_tol": rargs.aoc_gms_rel_tol,
    "aoc_gms_n_angles": rargs.aoc_gms_n_angles,
    "aoc_gms_cube_trick": rargs.aoc_gms_cube_trick,
    "aoc_gms_pc_cache_enable": rargs.aoc_gms_pc_cache_enable,
    "aoc_gms_upper_bound_tol_rt": rargs.aoc_gms_upper_bound_tol_rt,
    "aoc_gms_autofix_sampling": rargs.aoc_gms_autofix_sampling,
}
reward_fn = get_reward_function(
    failure_reward=rargs.failure_reward,
    iou_coef=rargs.iou_coef,
    cd_coef=rargs.cd_coef,
    auc_coef=rargs.auc_coef,
    aoc_gms_coef=rargs.aoc_gms_coef,
    mae_coef=rargs.mae_coef,
    nc_params=nc_params,
    mode=rargs.r_mode,
    print_every=rargs.print_sample_steps,
)


# those parameters will be passed to vllm generation trainer
bad_words = ["<|image_pad|>", "<|vision_pad|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>"]
ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in bad_words if processor.tokenizer.convert_tokens_to_ids(t) != processor.tokenizer.unk_token_id]
grpo.generation_kwargs = { 
    "bad_words":bad_words,
    "stop_token_ids": ids
}

# we override the steps_per_generation to generate all N samples at once per prompt and not be limited by batch size
# because we custom repeat the samples 
grpo.steps_per_generation = grpo.gradient_accumulation_steps
grpo.max_steps = grpo.num_train_epochs * len(hf_dataset) * grpo.num_iterations // (grpo.gradient_accumulation_steps*grpo.per_device_train_batch_size * int(os.environ["WORLD_SIZE"]))
print(f"Total training steps : {grpo.max_steps} ")
# we train based on steps not epochs
grpo.num_train_epochs = 0

trainer = TopSampleGRPOTrainer(
    clip_cov=targs.clip_cov,
    top_samples=targs.top_samples,
    model=model,
    processing_class=processor,
    reward_funcs=[reward_fn],
    train_dataset=hf_dataset,
    args=grpo,
    optimizers=(optimizer, lr_scheduler),
)

if targs.resume_ckpt_path == "" or targs.resume_ckpt_path == "False":
    resume_from_checkpoint = False
else :
    resume_from_checkpoint = targs.resume_ckpt_path
    print(f"resuming from checkpoint {resume_from_checkpoint}")
trainer.train(
    resume_from_checkpoint=resume_from_checkpoint
)

close_pool()
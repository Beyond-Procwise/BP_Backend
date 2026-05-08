import json,logging,os,sys,torch,transformers
os.environ["PYTHONPATH"]="src:."
sys.path.insert(0,"src")
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler("logs/finetune_native_agentnick.log",mode="w")])
L=logging.getLogger("ft")
L.info("=== NATIVE AgentNick — Gemma 4 26B (CPU offload) ===")
L.info("transformers: %s",transformers.__version__)

from datasets import Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments
from peft import LoraConfig,get_peft_model,TaskType,prepare_model_for_kbit_training
from trl import SFTTrainer

exs=[json.loads(l) for l in open("data/training/final_agentnick.jsonl") if l.strip()]
L.info("Dataset: %d examples",len(exs))

def fmt(ex):
    t=""
    for m in ex.get("messages",[]):
        if m["role"]=="user":t+="<start_of_turn>user\n"+m["content"]+"<end_of_turn>\n"
        elif m["role"]=="assistant":t+="<start_of_turn>model\n"+m["content"]+"<end_of_turn>\n"
    return{"text":t}

ds=Dataset.from_list(exs).map(fmt)

bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

L.info("Loading google/gemma-4-26B-A4B-it in 4-bit QLoRA...")
tok=AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it",trust_remote_code=True)
if not tok.pad_token:
    tok.pad_token=tok.eos_token

mod=AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-26B-A4B-it",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
L.info("Model loaded")

mod.enable_input_require_grads()
lora=LoraConfig(
    task_type=TaskType.CAUSAL_LM,r=32,lora_alpha=64,lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
mod=get_peft_model(mod,lora)
mod.print_trainable_parameters()

args=TrainingArguments(
    output_dir="data/models/native_agentnick",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=20,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=3,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
)

L.info("Training: 10 epochs, 324 examples, r=32, alpha=64")
tr=SFTTrainer(model=mod,train_dataset=ds,args=args,tokenizer=tok,dataset_text_field="text",max_seq_length=2048)
tr.train()
L.info("=== TRAINING COMPLETE ===")
tr.save_model("data/models/native_agentnick/final")
tok.save_pretrained("data/models/native_agentnick/final")
L.info("Saved to data/models/native_agentnick/final")

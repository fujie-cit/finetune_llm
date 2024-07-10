# %%
from datasets import Dataset, DatasetDict

dataset_dict = DatasetDict.load_from_disk('dataset_tokenized')


# %%
# 基本パラメータ
model_name = "cyberagent/open-calm-7b"
peft_name  = "lora-calm-7b"
output_dir = "lora-calm-7b-results"

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# %%
from transformers import AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer=AutoTokenizer.from_pretrained(model_name)
# model=AutoModelForCausalLM.from_pretrained(model_name)

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_8bit=True,
    quantization_config=bnb_config,
    device_map="auto",
)
model.gradient_checkpointing_enable()


# %%
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from peft import prepare_model_for_kbit_training

# モデルの前処理
# model = prepare_model_for_int8_training(model)
model = prepare_model_for_kbit_training(model)

# LoRAのパラメータ
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    fan_in_fan_out=False,
)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)

# 学習可能パラメータの確認
model.print_trainable_parameters()

# %%
import transformers
eval_steps = 10 # 2000
save_steps = 10 # 2000
logging_steps = 5 # 200

# トレーナーの準備
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["valid"],
    args = transformers.TrainingArguments(
        num_train_epochs=10,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        report_to="none",
        save_total_limit=3,
        push_to_hub=False,
        auto_find_batch_size=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# %%
# 学習の実行
model.config.use_cache = False
trainer.train() 
model.config.use_cache = True

# LoRAモデルの保存
trainer.model.save_pretrained(peft_name)



# %%
from datasets import Dataset, DatasetDict

dataset_dict = DatasetDict.load_from_disk('dataset_tokenized')


# %%
# 基本パラメータ
model_name = "cyberagent/open-calm-3b"
peft_name  = "lora-calm-3b"
output_dir = "lora-calm-3b-results"

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name)

# %%
from transformers import AutoModelForCausalLM

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

# %%
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# LoRAのパラメータ
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# モデルの前処理
model = prepare_model_for_int8_training(model)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)

# 学習可能パラメータの確認
model.print_trainable_parameters()

# %%
import transformers
eval_steps = 2000
save_steps = 2000
logging_steps = 200

# トレーナーの準備
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["valid"],
    args=transformers.TrainingArguments(
        num_train_epochs=3,
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
        auto_find_batch_size=True
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



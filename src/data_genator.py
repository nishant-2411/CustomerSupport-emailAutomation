"""
Shipcube AI - LoRA Fine-Tuning Script (Production Ready)
Flexible training pipeline for custom email processing datasets
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Configure logging with file and console handlers."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("shipcube-lora")
    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ShipcubeConfig:
    """Configuration for Shipcube LoRA training."""
    
    # Model settings
    base_model: str = "iprajwaal/gemma-3b-chat-support"
    use_4bit: bool = True
    use_8bit: bool = False
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Data settings
    max_length: int = 768
    val_split: float = 0.1
    
    # Training settings
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    
    # Optimization
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Output
    output_dir: str = "./checkpoints/shipcube-lora"
    save_total_limit: int = 3
    
    # Prompt template
    system_prompt: str = """You are Shipcube Assistant, an AI agent for Shipcube - a shipping and logistics platform.
Analyze the customer email and respond with a JSON object containing:
- query: Concise summary of customer's request
- intent: Primary intent category
- sentiment: Customer's emotional state (frustrated/neutral/satisfied/urgent)
- priority: Urgency level (critical/high/medium/low)
- entities: Extracted data (tracking_number, order_id, address, etc.)
- response: Professional draft response

Respond ONLY with valid JSON, no additional text."""

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_json(cls, path: str) -> "ShipcubeConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# ============================================================================
# DATA PROCESSING
# ============================================================================

class ShipcubeDataProcessor:
    """Flexible data processor for various dataset formats."""
    
    def __init__(self, config: ShipcubeConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.stats = {"total": 0, "valid": 0, "skipped": 0, "errors": []}
    
    def detect_format(self, df: pd.DataFrame) -> str:
        """Auto-detect dataset format based on columns."""
        columns = set(df.columns.str.lower())
        
        # Format 1: Full structured (email + all output fields)
        if all(c in columns for c in ["email", "intent", "response"]):
            return "structured"
        
        # Format 2: Simple pairs (input/output or email/response)
        if {"input", "output"}.issubset(columns) or {"email", "response"}.issubset(columns):
            return "pairs"
        
        # Format 3: Conversation format (messages/conversations)
        if "messages" in columns or "conversations" in columns:
            return "conversation"
        
        # Format 4: Text only (single column for completion)
        if "text" in columns:
            return "text"
        
        # Format 5: Instruction format (instruction, input, output)
        if {"instruction", "output"}.issubset(columns):
            return "instruction"
        
        raise ValueError(f"Unknown dataset format. Columns: {list(df.columns)}")
    
    def create_prompt(self, email: str) -> str:
        """Create formatted prompt from email content."""
        return f"""{self.config.system_prompt}

---
EMAIL:
\"\"\"
{email.strip()}
\"\"\"

---
JSON RESPONSE:"""

    def process_structured(self, row: pd.Series) -> Optional[Dict]:
        """Process structured format with all fields."""
        try:
            email = str(row.get("email", row.get("Email", ""))).strip()
            if not email:
                return None
            
            # Build response object from available columns
            response_obj = {}
            
            # Map common column variations
            field_mapping = {
                "query": ["query", "summary", "request"],
                "intent": ["intent", "category", "type"],
                "sentiment": ["sentiment", "emotion", "tone"],
                "priority": ["priority", "urgency", "level"],
                "response": ["response", "reply", "answer", "draft"],
            }
            
            for field, variants in field_mapping.items():
                for variant in variants:
                    # Check both lowercase and original case
                    for col in [variant, variant.lower(), variant.title()]:
                        if col in row.index and pd.notna(row[col]):
                            response_obj[field] = str(row[col]).strip()
                            break
                    if field in response_obj:
                        break
            
            # Handle entities (could be JSON string or dict)
            entities_raw = row.get("entities", row.get("Entities", "{}"))
            if isinstance(entities_raw, str):
                try:
                    response_obj["entities"] = json.loads(entities_raw) if entities_raw else {}
                except json.JSONDecodeError:
                    response_obj["entities"] = {}
            elif isinstance(entities_raw, dict):
                response_obj["entities"] = entities_raw
            else:
                response_obj["entities"] = {}
            
            prompt = self.create_prompt(email)
            completion = json.dumps(response_obj, indent=2)
            
            return {
                "prompt": prompt,
                "completion": completion,
                "text": f"{prompt}\n{completion}",
                "intent": response_obj.get("intent", "unknown"),
            }
            
        except Exception as e:
            self.stats["errors"].append(f"Row processing error: {e}")
            return None
    
    def process_pairs(self, row: pd.Series) -> Optional[Dict]:
        """Process simple input/output pairs."""
        try:
            # Get input (email)
            email = str(row.get("input", row.get("email", row.get("Input", row.get("Email", ""))))).strip()
            # Get output (response)
            output = str(row.get("output", row.get("response", row.get("Output", row.get("Response", ""))))).strip()
            
            if not email or not output:
                return None
            
            prompt = self.create_prompt(email)
            
            # Check if output is already JSON
            try:
                parsed = json.loads(output)
                completion = json.dumps(parsed, indent=2)
                intent = parsed.get("intent", "unknown")
            except json.JSONDecodeError:
                # Wrap plain text response in JSON structure
                completion = json.dumps({
                    "query": "Customer inquiry",
                    "intent": "general_inquiry",
                    "sentiment": "neutral",
                    "priority": "medium",
                    "entities": {},
                    "response": output
                }, indent=2)
                intent = "general_inquiry"
            
            return {
                "prompt": prompt,
                "completion": completion,
                "text": f"{prompt}\n{completion}",
                "intent": intent,
            }
            
        except Exception as e:
            self.stats["errors"].append(f"Pair processing error: {e}")
            return None
    
    def process_instruction(self, row: pd.Series) -> Optional[Dict]:
        """Process instruction/input/output format."""
        try:
            instruction = str(row.get("instruction", "")).strip()
            input_text = str(row.get("input", "")).strip()
            output = str(row.get("output", "")).strip()
            
            # Combine instruction and input as the email context
            email = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
            
            if not email or not output:
                return None
            
            prompt = self.create_prompt(email)
            
            try:
                parsed = json.loads(output)
                completion = json.dumps(parsed, indent=2)
                intent = parsed.get("intent", "unknown")
            except json.JSONDecodeError:
                completion = json.dumps({
                    "query": instruction[:100],
                    "intent": "general_inquiry",
                    "sentiment": "neutral",
                    "priority": "medium",
                    "entities": {},
                    "response": output
                }, indent=2)
                intent = "general_inquiry"
            
            return {
                "prompt": prompt,
                "completion": completion,
                "text": f"{prompt}\n{completion}",
                "intent": intent,
            }
            
        except Exception as e:
            self.stats["errors"].append(f"Instruction processing error: {e}")
            return None
    
    def process_text(self, row: pd.Series) -> Optional[Dict]:
        """Process pre-formatted text data."""
        try:
            text = str(row.get("text", "")).strip()
            if not text:
                return None
            
            return {
                "prompt": "",
                "completion": "",
                "text": text,
                "intent": "unknown",
            }
        except Exception as e:
            self.stats["errors"].append(f"Text processing error: {e}")
            return None
    
    def load_dataset(
        self, 
        path: str, 
        format_override: Optional[str] = None
    ) -> Tuple[List[Dict], Dict]:
        """Load and process dataset from file."""
        self.logger.info(f"Loading dataset from {path}")
        
        # Support multiple file formats
        path = Path(path)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in [".json", ".jsonl"]:
            df = pd.read_json(path, lines=path.suffix == ".jsonl")
        elif path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self.logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Detect or use override format
        data_format = format_override or self.detect_format(df)
        self.logger.info(f"Using data format: {data_format}")
        
        # Select processor
        processors = {
            "structured": self.process_structured,
            "pairs": self.process_pairs,
            "instruction": self.process_instruction,
            "text": self.process_text,
        }
        
        processor = processors.get(data_format)
        if not processor:
            raise ValueError(f"No processor for format: {data_format}")
        
        # Process all rows
        examples = []
        self.stats = {"total": len(df), "valid": 0, "skipped": 0, "errors": []}
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            result = processor(row)
            if result:
                examples.append(result)
                self.stats["valid"] += 1
            else:
                self.stats["skipped"] += 1
        
        self.logger.info(
            f"Processed: {self.stats['valid']} valid, "
            f"{self.stats['skipped']} skipped, "
            f"{len(self.stats['errors'])} errors"
        )
        
        if self.stats["errors"]:
            self.logger.warning(f"Sample errors: {self.stats['errors'][:5]}")
        
        return examples, self.stats
    
    def split_dataset(
        self, 
        data: List[Dict], 
        val_split: float = 0.1,
        stratify: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train/validation sets."""
        if len(data) < 10:
            self.logger.warning("Dataset too small for splitting, using all for training")
            return data, data[:2]
        
        if stratify:
            try:
                intents = [d.get("intent", "unknown") for d in data]
                # Only stratify if we have enough samples per class
                intent_counts = pd.Series(intents).value_counts()
                min_samples = intent_counts.min()
                
                if min_samples >= 2:
                    train_data, val_data = train_test_split(
                        data, test_size=val_split, stratify=intents, random_state=42
                    )
                else:
                    self.logger.warning("Not enough samples for stratification, using random split")
                    train_data, val_data = train_test_split(
                        data, test_size=val_split, random_state=42
                    )
            except Exception as e:
                self.logger.warning(f"Stratification failed: {e}, using random split")
                train_data, val_data = train_test_split(
                    data, test_size=val_split, random_state=42
                )
        else:
            train_data, val_data = train_test_split(
                data, test_size=val_split, random_state=42
            )
        
        self.logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data


# ============================================================================
# MODEL SETUP
# ============================================================================

class ShipcubeModelManager:
    """Manage model loading, LoRA setup, and quantization."""
    
    def __init__(self, config: ShipcubeConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
    
    def _get_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            device = "mps"
            self.logger.info("Using Apple MPS")
        else:
            device = "cpu"
            self.logger.info("Using CPU (training will be slow)")
        return device
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config for memory efficiency."""
        if self.config.use_4bit and self.device == "cuda":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit and self.device == "cuda":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def load_model(self) -> Tuple[Any, Any]:
        """Load base model and tokenizer with optional quantization."""
        self.logger.info(f"Loading model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
        }
        
        # Add quantization if available
        quant_config = self._get_quantization_config()
        if quant_config:
            model_kwargs["quantization_config"] = quant_config
            self.logger.info("Using 4-bit quantization")
        
        # Device mapping
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs
        )
        
        # Prepare for k-bit training if quantized
        if quant_config:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        elif self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.logger.info(f"Model loaded: {self.model.config.model_type}")
        return self.model, self.tokenizer
    
    def apply_lora(self) -> Any:
        """Apply LoRA adapters to the model."""
        self.logger.info("Applying LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            inference_mode=False,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Log trainable parameters
        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total
        self.logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")
        
        return self.model
    
    def save_model(self, path: str):
        """Save LoRA adapters and tokenizer."""
        self.logger.info(f"Saving model to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save(os.path.join(path, "training_config.json"))


# ============================================================================
# TRAINING CALLBACKS
# ============================================================================

class ShipcubeLoggingCallback(TrainerCallback):
    """Custom callback for detailed training logs."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.best_loss = float("inf")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", logs.get("eval_loss"))
            if loss and loss < self.best_loss:
                self.best_loss = loss
            
            step_info = f"Step {state.global_step}"
            metrics = " | ".join([f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))])
            self.logger.info(f"{step_info} | {metrics}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.logger.info(f"Epoch {state.epoch:.0f} completed | Best loss: {self.best_loss:.4f}")


# ============================================================================
# TRAINER
# ============================================================================

class ShipcubeTrainer:
    """Main trainer class orchestrating the training pipeline."""
    
    def __init__(self, config: ShipcubeConfig):
        self.config = config
        self.logger = setup_logging(config.output_dir)
        self.data_processor = ShipcubeDataProcessor(config, self.logger)
        self.model_manager = ShipcubeModelManager(config, self.logger)
    
    def prepare_datasets(
        self, 
        train_data: List[Dict], 
        val_data: List[Dict]
    ) -> Tuple[Dataset, Dataset]:
        """Tokenize and prepare datasets for training."""
        
        def tokenize(examples):
            tokens = self.model_manager.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
                return_tensors=None,
            )
            # Set labels for language modeling
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Tokenize
        remove_cols = ["prompt", "completion", "text", "intent"]
        train_dataset = train_dataset.map(
            tokenize, 
            batched=True, 
            remove_columns=[c for c in remove_cols if c in train_dataset.column_names],
            desc="Tokenizing train"
        )
        val_dataset = val_dataset.map(
            tokenize, 
            batched=True, 
            remove_columns=[c for c in remove_cols if c in val_dataset.column_names],
            desc="Tokenizing val"
        )
        
        self.logger.info(f"Train dataset: {len(train_dataset)} samples")
        self.logger.info(f"Val dataset: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> Dict:
        """Run the training loop."""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler,
            fp16=self.config.fp16 and self.model_manager.device == "cuda",
            bf16=self.config.bf16 and self.model_manager.device == "cuda",
            logging_steps=10,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            optim="adamw_torch",
            max_grad_norm=1.0,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_manager.tokenizer,
            mlm=False,
        )
        
        # Callbacks
        callbacks = [
            ShipcubeLoggingCallback(self.logger),
            EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            ),
        ]
        
        # Trainer
        trainer = Trainer(
            model=self.model_manager.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Train
        self.logger.info("=" * 60)
        self.logger.info("Starting training")
        self.logger.info("=" * 60)
        
        train_result = trainer.train()
        
        # Final evaluation
        eval_result = trainer.evaluate()
        
        # Save final model
        self.model_manager.save_model(self.config.output_dir)
        
        # Compile metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result["eval_loss"],
            "train_runtime_seconds": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "total_steps": train_result.global_step,
            "epochs_completed": train_result.metrics.get("epoch", self.config.epochs),
        }
        
        # Save metrics
        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info("=" * 60)
        self.logger.info("Training completed!")
        self.logger.info(f"Final train loss: {metrics['train_loss']:.4f}")
        self.logger.info(f"Final eval loss: {metrics['eval_loss']:.4f}")
        self.logger.info(f"Model saved to: {self.config.output_dir}")
        self.logger.info("=" * 60)
        
        return metrics
    
    def run(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        format_override: Optional[str] = None,
    ) -> Dict:
        """Run complete training pipeline."""
        
        # Load and process data
        train_data, train_stats = self.data_processor.load_dataset(train_file, format_override)
        
        if val_file:
            val_data, val_stats = self.data_processor.load_dataset(val_file, format_override)
        else:
            train_data, val_data = self.data_processor.split_dataset(
                train_data, 
                self.config.val_split
            )
        
        # Load model
        self.model_manager.load_model()
        self.model_manager.apply_lora()
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets(train_data, val_data)
        
        # Train
        metrics = self.train(train_dataset, val_dataset)
        
        return metrics


# ============================================================================
# INFERENCE
# ============================================================================

class ShipcubeInference:
    """Load trained model and run inference."""
    
    def __init__(self, model_path: str, base_model: Optional[str] = None):
        self.model_path = model_path
        
        # Load config
        config_path = os.path.join(model_path, "training_config.json")
        if os.path.exists(config_path):
            self.config = ShipcubeConfig.from_json(config_path)
        else:
            self.config = ShipcubeConfig()
        
        if base_model:
            self.config.base_model = base_model
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
    
    def process_email(
        self, 
        email: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> Dict:
        """Process a single email and return structured response."""
        
        # Create prompt
        prompt = f"""{self.config.system_prompt}

---
EMAIL:
\"\"\"
{email.strip()}
\"\"\"

---
JSON RESPONSE:"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            # Find JSON in response
            json_start = response_text.rfind("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        return {"error": "Failed to parse response", "raw": response_text}
    
    def batch_process(
        self, 
        emails: List[str], 
        show_progress: bool = True
    ) -> List[Dict]:
        """Process multiple emails."""
        results = []
        iterator = tqdm(emails, desc="Processing emails") if show_progress else emails
        
        for email in iterator:
            result = self.process_email(email)
            results.append(result)
        
        return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Shipcube AI - LoRA Fine-Tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--train_file", type=str, required=True, help="Training data file")
    train_parser.add_argument("--val_file", type=str, default=None, help="Validation data file (optional)")
    train_parser.add_argument("--output_dir", type=str, default="./checkpoints/shipcube-lora")
    train_parser.add_argument("--base_model", type=str, default="iprajwaal/gemma-3b-chat-support")
    train_parser.add_argument("--format", type=str, choices=["structured", "pairs", "instruction", "text"], 
                              default=None, help="Override data format detection")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--learning_rate", type=float, default=2e-4)
    train_parser.add_argument("--lora_r", type=int, default=16)
    train_parser.add_argument("--lora_alpha", type=int, default=32)
    train_parser.add_argument("--max_length", type=int, default=768)
    train_parser.add_argument("--val_split", type=float, default=0.1)
    train_parser.add_argument("--use_4bit", action="store_true", default=True)
    train_parser.add_argument("--no_4bit", action="store_false", dest="use_4bit")
    train_parser.add_argument("--config", type=str, default=None, help="Load config from JSON file")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    infer_parser.add_argument("--email", type=str, default=None, help="Single email to process")
    infer_parser.add_argument("--input_file", type=str, default=None, help="File with emails to process")
    infer_parser.add_argument("--output_file", type=str, default=None, help="Output file for results")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--data_file", type=str, required=True, help="Data file to validate")
    validate_parser.add_argument("--format", type=str, default=None, help="Override format detection")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # Build config
        if args.config:
            config = ShipcubeConfig.from_json(args.config)
        else:
            config = ShipcubeConfig(
                base_model=args.base_model,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                max_length=args.max_length,
                val_split=args.val_split,
                use_4bit=args.use_4bit,
            )
        
        # Run training
        trainer = ShipcubeTrainer(config)
        metrics = trainer.run(
            train_file=args.train_file,
            val_file=args.val_file,
            format_override=args.format,
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(json.dumps(metrics, indent=2))
    
    elif args.command == "infer":
        inference = ShipcubeInference(args.model_path)
        
        if args.email:
            result = inference.process_email(args.email)
            print(json.dumps(result, indent=2))
        
        elif args.input_file:
            # Load emails from file
            with open(args.input_file) as f:
                if args.input_file.endswith(".json"):
                    emails = json.load(f)
                else:
                    emails = [line.strip() for line in f if line.strip()]
            
            results = inference.batch_process(emails)
            
            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {args.output_file}")
            else:
                print(json.dumps(results, indent=2))
        
        else:
            print("Provide --email or --input_file")
    
    elif args.command == "validate":
        config = ShipcubeConfig()
        logger = setup_logging("./logs")
        processor = ShipcubeDataProcessor(config, logger)
        
        data, stats = processor.load_dataset(args.data_file, args.format)
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"Total rows: {stats['total']}")
        print(f"Valid samples: {stats['valid']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Errors: {len(stats['errors'])}")
        
        if data:
            print(f"\nSample processed entry:")
            print(json.dumps(data[0], indent=2)[:500] + "...")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
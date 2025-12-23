"""
Evaluation Script for Shipcube AI Email Processing Model
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score


def load_model_for_inference(model_path: str, adapter_path: str = None):
    """Load model for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, email: str) -> str:
    """Generate response for an email."""
    prompt = f"""You are Shipcube Assistant. Read the email and produce JSON:
{{ "query": "...", "intent":"...", "sentiment":"...", "priority":"...", "entities": {{...}}, "response":"..." }}
Email:
\"\"\"{email}\"\"\""""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract JSON from response
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return response[json_start:json_end]
    except:
        pass
    
    return response


def parse_prediction(response: str) -> Dict[str, Any]:
    """Parse prediction from model response."""
    try:
        return json.loads(response)
    except:
        return {}


def evaluate(model, tokenizer, test_file: str) -> Dict[str, float]:
    """Evaluate model on test set."""
    df = pd.read_csv(test_file)
    
    predictions = []
    ground_truth = []
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    intent_true = []
    intent_pred = []
    sentiment_true = []
    sentiment_pred = []
    rouge_scores = []
    
    for idx, row in df.iterrows():
        email = row['email']
        gt = {
            "query": row.get("query", ""),
            "intent": row.get("intent", ""),
            "sentiment": row.get("sentiment", ""),
            "priority": row.get("priority", ""),
            "entities": json.loads(row.get("entities", "{}")),
            "response": row.get("response", "")
        }
        
        # Generate prediction
        pred_response = generate_response(model, tokenizer, email)
        pred = parse_prediction(pred_response)
        
        # Collect for metrics
        intent_true.append(gt.get("intent", ""))
        intent_pred.append(pred.get("intent", ""))
        sentiment_true.append(gt.get("sentiment", ""))
        sentiment_pred.append(pred.get("sentiment", ""))
        
        # ROUGE score for response
        gt_response = gt.get("response", "")
        pred_response_text = pred.get("response", "")
        rouge_scores.append(rouge.score(gt_response, pred_response_text))
        
        predictions.append(pred)
        ground_truth.append(gt)
    
    # Calculate metrics
    intent_acc = accuracy_score(intent_true, intent_pred)
    intent_f1 = f1_score(intent_true, intent_pred, average='weighted', zero_division=0)
    sentiment_acc = accuracy_score(sentiment_true, sentiment_pred)
    
    avg_rouge1 = sum([s['rouge1'].fmeasure for s in rouge_scores]) / len(rouge_scores)
    avg_rougel = sum([s['rougeL'].fmeasure for s in rouge_scores]) / len(rouge_scores)
    
    results = {
        "intent_accuracy": intent_acc,
        "intent_f1": intent_f1,
        "sentiment_accuracy": sentiment_acc,
        "rouge1": avg_rouge1,
        "rougeL": avg_rougel
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shipcube AI model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--test_file", type=str, default="../data/shipcube_test.csv", help="Test CSV file")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model_for_inference(args.model_path, args.adapter_path)
    print("Model loaded successfully!")
    
    # Evaluate
    print("Running evaluation...")
    results = evaluate(model, tokenizer, args.test_file)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("="*50 + "\n")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()


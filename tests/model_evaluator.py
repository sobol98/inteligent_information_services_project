import torch
import numpy as np
from tqdm import tqdm
import json
from typing import Dict
import nltk
from nltk.translate.bleu_score import sentence_bleu
import asyncio
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import ModelManager

class ModelEvaluator:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        
        # Stałe przykładowe teksty do ewaluacji
        self.test_pairs = [
            ("The book is about", "The book is about a young wizard who discovers he has magical powers and goes to a special school."),
            ("The movie tells the story", "The movie tells the story of a group of friends who embark on an adventure across the galaxy."),
            ("In the future", "In the future, humanity has colonized Mars and established the first permanent settlement."),
            ("Scientists discovered", "Scientists discovered a new species of deep-sea creatures living near hydrothermal vents."),
            ("The ancient civilization", "The ancient civilization developed advanced mathematical systems and built impressive structures.")
        ]

    def calculate_perplexity(self, input_text: str) -> float:
        """Oblicza perplexity dla tekstu"""
        try:
            inputs = self.model_manager.tokenizer(
                input_text,
                return_tensors="pt"
            ).to(self.model_manager.device)
            
            with torch.no_grad():
                outputs = self.model_manager.model(**inputs)
            
            return torch.exp(outputs.loss).item()
        except Exception as e:
            print(f"Perplexity calculation error: {e}")
            return float('inf')

    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Oblicza BLEU score"""
        try:
            reference_tokens = reference.lower().split()
            hypothesis_tokens = hypothesis.lower().split()
            return sentence_bleu([reference_tokens], hypothesis_tokens)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return 0.0

    async def evaluate_model(self) -> Dict:
        results = {
            "model_name": self.model_manager.model_name,
            "samples": [],
            "perplexity_scores": [],
            "bleu_scores": []
        }

        for prompt, reference in tqdm(self.test_pairs, desc="Evaluating"):
            # Generacja tekstu
            prediction = await self.model_manager.predict_batch([prompt])
            generated_text = prediction[0] if prediction else ""
            
            # Obliczanie metryk
            perplexity = self.calculate_perplexity(generated_text)
            bleu = self.calculate_bleu(reference, generated_text)
            
            results["samples"].append({
                "prompt": prompt,
                "reference": reference,
                "generated": generated_text,
                "perplexity": perplexity,
                "bleu": bleu
            })
            
            results["perplexity_scores"].append(perplexity)
            results["bleu_scores"].append(bleu)

        # Średnie wyniki
        results["avg_perplexity"] = np.mean(results["perplexity_scores"])
        results["avg_bleu"] = np.mean(results["bleu_scores"])

        return results

async def main():
    models = {
        1: 'distilgpt2',
        2: 'gpt2'
    }

    for model_id, model_name in models.items():
        print(f"\nEvaluating model: {model_name}")
        
        model_manager = ModelManager(model_name)
        try:
            await model_manager.load_model()
            await model_manager.start_processing()
            
            evaluator = ModelEvaluator(model_manager)
            results = await evaluator.evaluate_model()

            print(f"\nResults for {model_name}:")
            print(f"Average Perplexity: {results['avg_perplexity']:.2f}")
            print(f"Average BLEU: {results['avg_bleu']:.4f}")

            print("\nSample Generations:")
            for sample in results["samples"][:2]:
                print(f"\nPrompt: {sample['prompt']}")
                print(f"Generated: {sample['generated']}")
                print(f"Reference: {sample['reference']}")
                print(f"Perplexity: {sample['perplexity']:.2f}")
                print(f"BLEU: {sample['bleu']:.4f}")

            filename = f"eval_results_{model_name.replace('/', '_')}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)

        finally:
            await model_manager.stop_processing()
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
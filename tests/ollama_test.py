import asyncio
import aiohttp
import time
import statistics
import json
import subprocess
import re
import psutil
import GPUtil

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.models = {
            1: "llama3.2:1b",
            2: "falcon3:1b",
            3: "qwen2:1.5b",
            4: "mapler/gpt2:latest"
        }
        self.current_model = self.models[1]
        self.response_times = []
        self.vram_measurements = []

    def _get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Bierzemy pierwszą kartę GPU (zakładając, że Ollama używa głównej karty)
                gpu = gpus[0]
                return {
                    'total': gpu.memoryTotal,
                    'used': gpu.memoryUsed,
                    'free': gpu.memoryFree
                }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
        return None

    async def predict(self, text: str):
        """Send prediction request to Ollama"""
        async with aiohttp.ClientSession() as session:
            # Pomiar VRAM przed
            vram_before = self._get_gpu_memory_usage()
            start_time = time.time()
            
            payload = {
                "model": self.current_model,
                "prompt": text,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "top_k": 50,
                    "repeat_penalty": 1.1,
                    "num_predict": 100,
                }
            }
            
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Request failed with status {response.status}")
                
                result = await response.json()
                end_time = time.time()
                
                # Pomiar VRAM po
                vram_after = self._get_gpu_memory_usage()
                
                response_time = end_time - start_time
                self.response_times.append(response_time)
                
                # Obliczanie delty VRAM
                vram_delta = 0
                if vram_before and vram_after:
                    vram_delta = vram_after['used'] - vram_before['used']
                    self.vram_measurements.append(vram_delta)
                
                return {
                    "text": text,
                    "prediction": result['response'],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S:%f")[:-3],
                    "response_time": response_time,
                    "vram_delta": vram_delta,
                    "vram_total": vram_after['total'] if vram_after else None,
                    "vram_used": vram_after['used'] if vram_after else None
                }

    async def test_predictions(self, texts: list, concurrency: int = 10, model_id: int = 1):
        """Test predictions with VRAM monitoring"""
        self.current_model = self.models[model_id]
        self.response_times = []
        self.vram_measurements = []
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_predict(text):
            async with semaphore:
                return await self.predict(text)
        
        # Pomiar początkowy VRAM
        initial_vram = self._get_gpu_memory_usage()
        
        start_time = time.time()
        results = await asyncio.gather(*[limited_predict(text) for text in texts])
        end_time = time.time()
        
        # Pomiar końcowy VRAM
        final_vram = self._get_gpu_memory_usage()
        
        return {
            "model": self.current_model,
            "results": results,
            "total_time": end_time - start_time,
            "avg_time_per_request": (end_time - start_time) / len(texts),
            "response_times": self.response_times,
            "vram_measurements": self.vram_measurements,
            "initial_vram": initial_vram,
            "final_vram": final_vram
        }

async def main():
    client = OllamaClient()
    
    test_texts = [
        "Once upon a time",
        "In a world where",
        "The future of technology",
        "Machine learning is",
        "Artificial intelligence will",
        "Quantum computing is",
        "Climate change is",
        "Renewable energy is",
        "The history of the world",
        'My mother is a doctor',
        'My father is a teacher',
        'One day I bought a red',
        'Lisbon is the capital of',
        'Washington is the capital of',
        'The capital of France is',
        'Ancient Rome was a',
        'Print function in Python is used to',
        'The most popular programming language is',
        'Graphs are used to represent',
        'The first Newton law states that',
        'The second Newton law states that',
        'Leonardo Da Vinci live in'
    ]
    
    try:
        # 1 - Llama 3.2:1B
        # 2 - Falcon 3:1B
        # 3 - Qwen 2:1.5B
        # 4 - GPT-2
        
        model_id = 4
    
        print(f"\nTesting model: {client.models[model_id]}")
        
        # Czekamy na ustabilizowanie się pamięci
        
        concurrent_results = await client.test_predictions(
            test_texts, 
            concurrency=10,
            model_id=model_id
        )
        
        # Detailed output
        print("\nDetailed Predictions:")
        for i, result in enumerate(concurrent_results['results'], 1):
            print(f"\nPrediction {i}:")
            print(f"Input: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Response Time: {result['response_time']:.4f}s")
            print(f"VRAM Delta: {result['vram_delta']:.2f} MB")
        
        # VRAM Statistics
        if concurrent_results['vram_measurements']:
            print("\nVRAM Usage Statistics:")
            print(f"Initial VRAM: {concurrent_results['initial_vram']['used']:.2f} MB")
            print(f"Final VRAM: {concurrent_results['final_vram']['used']:.2f} MB")
            print(f"Average VRAM Delta: {statistics.mean(concurrent_results['vram_measurements']):.2f} MB")
            print(f"Max VRAM Delta: {max(concurrent_results['vram_measurements']):.2f} MB")
            print(f"Min VRAM Delta: {min(concurrent_results['vram_measurements']):.2f} MB")
            if len(concurrent_results['vram_measurements']) > 1:
                print(f"VRAM Delta Std Dev: {statistics.stdev(concurrent_results['vram_measurements']):.2f} MB")
        
        # Performance Statistics
        print("\nPerformance Statistics:")
        print(f"Average Response Time: {statistics.mean(concurrent_results['response_times']):.4f}s")
        print(f"Min Response Time: {min(concurrent_results['response_times']):.4f}s")
        print(f"Max Response Time: {max(concurrent_results['response_times']):.4f}s")
        print(f"Response Time Std Dev: {statistics.stdev(concurrent_results['response_times']):.4f}s")
        
        print(f"\nTotal Requests: {len(concurrent_results['results'])}")
        print(f"Total Time: {concurrent_results['total_time']:.2f}s")
        print(f"Avg Time per Request: {concurrent_results['avg_time_per_request']:.4f}s")
        
        # Save results
        filename = f"ollama_results_{client.models[model_id].replace(':', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(concurrent_results, f, indent=2)
        print(f"\nResults saved to {filename}")
        
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
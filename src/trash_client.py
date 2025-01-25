import asyncio
import aiohttp
import time
import random
import torch
import statistics
import os
os.system('nvidia-smi')

class PredictionClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.vram_measurements = []
                
    async def predict(self, text: str):
        """Send prediction request to the service"""
        async with aiohttp.ClientSession() as session:
            
            vram_before = torch.cuda.memory_allocated() / 1024 / 1024
                        
            async with session.post(f"{self.base_url}/predict", json={"text": text}) as response:
                if response.status != 200:
                    raise Exception(f"Request failed with status {response.status}")
                
                vram_after = torch.cuda.memory_allocated() / 1024 / 1024
                self.vram_measurements.append(vram_after - vram_before)
                print(f"Memory cashed: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

                result = await response.json()
                result['vram_delta'] = vram_after - vram_before
                
                return result

    async def test_predictions(self, texts: list, concurrency: int = 10):
        """
        Send multiple prediction requests concurrently
        
        Args:
            texts (list): List of input texts to predict
            concurrency (int): Number of concurrent requests
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_predict(text):
            async with semaphore:
                return await self.predict(text)
        
        start_time = time.time()
        results = await asyncio.gather(*[limited_predict(text) for text in texts])
        end_time = time.time()
        
        return {
            "results": results,
            "total_time": end_time - start_time,
            "avg_time_per_request": (end_time - start_time) / len(texts),
            "vram_measurements": self.vram_measurements

        }

# Example usage script
async def main():
    client = PredictionClient()
    
    # Generate test texts
    test_texts = [
        "Once upon a time",
        "In a world where",
        "The future of technology",
        "Machine learning is",
        "Artificial intelligence will",
        "Quantum computing is",
        "Climate change is",
        "Renewable energy is"
    ]
    
    try:
        # Test single prediction
        print("Single Prediction Test:")
        single_result = await client.predict(test_texts[0])
        print(f"Input: {single_result['text']}")
        print(f"Prediction: {single_result['prediction']}")
        print(f"Timestamp: {single_result['timestamp']}")
        print(f"VRAM Delta: {single_result['vram_delta']:.2f} MB")
        
                
        # Test concurrent predictions
        print("\nConcurrent Predictions Test:")
        concurrent_results = await client.test_predictions(test_texts)
        
                # Detailed output for each prediction
        print("\nDetailed Predictions:")
        for i, result in enumerate(concurrent_results['results'], 1):
            print(f"\nPrediction {i}:")
            print(f"Input: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Timestamp: {result['timestamp']}")
        
        
        
        # VRAM Statistics
        
        vram_measurements = concurrent_results['vram_measurements']
        print("\nVRAM Usage Statistics:")
        print(f"Average VRAM Delta: {statistics.mean(vram_measurements):.2f} MB")
        print(f"Min VRAM Delta: {min(vram_measurements):.2f} MB")
        print(f"Max VRAM Delta: {max(vram_measurements):.2f} MB")
        
        print(f"Total Requests: {len(concurrent_results['results'])}")
        print(f"Total Time: {concurrent_results['total_time']:.2f}s")
        print(f"Avg Time per Request: {concurrent_results['avg_time_per_request']:.4f}s")
        
        
        print(f"Memory cashed: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
        print(f"Memory allocated: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
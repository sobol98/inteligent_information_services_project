import asyncio
import time
import random
from typing import List, Dict
from trash_client import PredictionClient

class LoadTester:
    def __init__(
        self, 
        base_url: str = "http://localhost:8000", 
        total_requests: int = 100, 
        concurrency: int = 10
    ):
        self.client = PredictionClient(base_url)
        self.total_requests = total_requests
        self.concurrency = concurrency
    
    def generate_test_data(self) -> List[str]:
        """Generate diverse test inputs"""
        prefixes = [
            "The future of", 
            "Once upon a", 
            "In a world", 
            "Machine learning", 
            "Artificial intelligence",
            "Quantum computing",
            "Climate change",
            "Renewable energy"
        ]
        return [random.choice(prefixes) for _ in range(self.total_requests)]
    
    async def run_load_test(self) -> Dict:
        """Perform comprehensive load testing"""
        test_inputs = self.generate_test_data()
        
        start_time = time.time()
        results = await self.client.test_predictions(
            test_inputs, 
            concurrency=self.concurrency
        )
        end_time = time.time()
        
        # Detailed metrics
        success_count = len(results['results'])
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": success_count,
            "total_time": end_time - start_time,
            "avg_time_per_request": results['avg_time_per_request'],
            "concurrency": self.concurrency
        }

async def main():
    # Different load test configurations
    configurations = [
        {"total_requests": 50, "concurrency": 5},
        {"total_requests": 100, "concurrency": 10},
        {"total_requests": 200, "concurrency": 20}
    ]
    
    for config in configurations:
        print(f"\nLoad Test Configuration: {config}")
        tester = LoadTester(**config)
        results = await tester.run_load_test()
        
        # Print results
        for key, value in results.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import aiounittest
from trash_client import PredictionClient
from trash_test import LoadTester
import unittest

class TestLoadTester(aiounittest.AsyncTestCase):
    def setUp(self):
        self.base_url = "http://localhost:8000"
        self.client = PredictionClient(self.base_url)
    
    async def test_generate_test_data(self):
        """Test test data generation"""
        tester = LoadTester(total_requests=10)
        test_data = tester.generate_test_data()
        
        self.assertEqual(len(test_data), 10)
        self.assertTrue(all(isinstance(item, str) for item in test_data))
    
    async def test_single_prediction(self):
        """Test single prediction"""
        input_text = "The future of"
        result = await self.client.predict(input_text)
        
        self.assertIn('text', result)
        self.assertIn('prediction', result)
        self.assertIn('timestamp', result)
        self.assertTrue(len(result['prediction']) > 0)
    
    async def test_concurrent_predictions(self):
        """Test concurrent predictions"""
        test_inputs = ["The future of"] * 20
        results = await self.client.test_predictions(test_inputs, concurrency=5)
        
        self.assertEqual(len(results['results']), 20)
        self.assertGreater(results['total_time'], 0)
    
    async def test_load_test_metrics(self):
        """Test load test configurations"""
        configs = [
            {"total_requests": 50, "concurrency": 5},
            {"total_requests": 100, "concurrency": 10}
        ]
        
        for config in configs:
            tester = LoadTester(base_url=self.base_url, **config)
            results = await tester.run_load_test()
            
            self.assertEqual(results['total_requests'], config['total_requests'])
            self.assertLessEqual(results['successful_requests'], config['total_requests'])
            self.assertGreater(results['total_time'], 0)
            self.assertGreater(results['avg_time_per_request'], 0)


if __name__ == '__main__':
    unittest.main()

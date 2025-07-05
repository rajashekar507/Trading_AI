"""
Performance Benchmark Suite for Trading_AI System
Tests all critical performance metrics
"""

import time
import psutil
import asyncio
import logging
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def benchmark_signal_generation(self):
        """Test signal generation speed"""
        try:
            from analysis.signal_engine import TradeSignalEngine
            from core.data_manager import DataManager
            
            print("🔍 Testing signal generation speed...")
            
            # Initialize components
            settings = {'confidence_threshold': 60}
            data_manager = DataManager(settings)
            signal_engine = TradeSignalEngine(settings, data_manager)
            
            # Benchmark signal generation
            start = time.time()
            
            # Generate signals for both instruments
            signals = []
            for instrument in ['NIFTY', 'BANKNIFTY']:
                try:
                    signal = signal_engine.generate_signals(instrument)
                    signals.extend(signal if signal else [])
                except Exception as e:
                    print(f"Signal generation error for {instrument}: {e}")
            
            end = time.time()
            
            self.results['signal_generation_time'] = end - start
            self.results['signals_generated'] = len(signals)
            
            print(f"✅ Signal generation: {end - start:.2f}s, Signals: {len(signals)}")
            
        except Exception as e:
            print(f"❌ Signal generation benchmark failed: {e}")
            self.results['signal_generation_time'] = None

    def benchmark_api_response(self):
        """Test API response times"""
        try:
            from data.market_data import MarketDataProvider
            
            print("🌐 Testing API response times...")
            
            provider = MarketDataProvider()
            
            # Test multiple API calls
            response_times = []
            
            for i in range(5):
                start = time.time()
                try:
                    # Test data fetch
                    data = provider.get_spot_data(['NIFTY 50', 'NIFTY BANK'])
                    end = time.time()
                    response_times.append(end - start)
                except Exception as e:
                    print(f"API call {i+1} failed: {e}")
                    
                time.sleep(1)  # Wait between calls
            
            if response_times:
                avg_response = sum(response_times) / len(response_times)
                self.results['avg_api_response_time'] = avg_response
                self.results['max_api_response_time'] = max(response_times)
                self.results['min_api_response_time'] = min(response_times)
                
                print(f"✅ API Response - Avg: {avg_response:.2f}s, Max: {max(response_times):.2f}s")
            else:
                print("❌ No successful API calls")
                
        except Exception as e:
            print(f"❌ API benchmark failed: {e}")
            self.results['avg_api_response_time'] = None

    def benchmark_system_resources(self):
        """Monitor system resource usage"""
        print("💻 Monitoring system resources...")
        
        # Get current process
        process = psutil.Process()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        
        self.results.update({
            'cpu_usage_percent': cpu_percent,
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': memory_percent,
            'system_memory_percent': system_memory.percent,
            'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100
        })
        
        print(f"✅ CPU: {cpu_percent}%, Memory: {memory_percent:.1f}%, Disk: {(disk_usage.used / disk_usage.total) * 100:.1f}%")

    def benchmark_data_processing(self):
        """Test data processing speed"""
        try:
            import pandas as pd
            import numpy as np
            
            print("📊 Testing data processing speed...")
            
            # Create sample data
            sample_size = 10000
            data = {
                'timestamp': pd.date_range('2025-01-01', periods=sample_size, freq='1min'),
                'open': np.random.uniform(25000, 26000, sample_size),
                'high': np.random.uniform(25000, 26000, sample_size),
                'low': np.random.uniform(25000, 26000, sample_size),
                'close': np.random.uniform(25000, 26000, sample_size),
                'volume': np.random.randint(1000, 10000, sample_size)
            }
            
            df = pd.DataFrame(data)
            
            # Benchmark data processing operations
            start = time.time()
            
            # Common operations
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['rsi'] = self.calculate_rsi(df['close'])
            df['volatility'] = df['close'].rolling(20).std()
            
            end = time.time()
            
            self.results['data_processing_time'] = end - start
            self.results['data_points_processed'] = len(df)
            
            print(f"✅ Data processing: {end - start:.2f}s for {len(df)} points")
            
        except Exception as e:
            print(f"❌ Data processing benchmark failed: {e}")
            self.results['data_processing_time'] = None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI for benchmark"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def run_full_benchmark(self):
        """Run complete performance benchmark suite"""
        print("🚀 Starting Trading_AI Performance Benchmark Suite")
        print("=" * 60)
        
        # Run all benchmarks
        self.benchmark_system_resources()
        self.benchmark_api_response()
        self.benchmark_signal_generation()
        self.benchmark_data_processing()
        
        # Calculate total benchmark time
        total_time = time.time() - self.start_time
        self.results['total_benchmark_time'] = total_time
        
        # Generate report
        self.generate_report()

    def generate_report(self):
        """Generate performance benchmark report"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        # System Performance
        print("💻 SYSTEM PERFORMANCE:")
        print(f"   CPU Usage: {self.results.get('cpu_usage_percent', 'N/A')}%")
        print(f"   Process Memory: {self.results.get('process_memory_mb', 'N/A'):.1f} MB")
        print(f"   Memory Usage: {self.results.get('process_memory_percent', 'N/A'):.1f}%")
        print(f"   System Memory: {self.results.get('system_memory_percent', 'N/A')}%")
        print(f"   Disk Usage: {self.results.get('disk_usage_percent', 'N/A'):.1f}%")
        
        # API Performance
        print("\n🌐 API PERFORMANCE:")
        if self.results.get('avg_api_response_time'):
            print(f"   Average Response: {self.results['avg_api_response_time']:.2f}s")
            print(f"   Max Response: {self.results['max_api_response_time']:.2f}s")
            print(f"   Min Response: {self.results['min_api_response_time']:.2f}s")
        else:
            print("   API tests failed")
        
        # Signal Generation
        print("\n⚡ SIGNAL GENERATION:")
        if self.results.get('signal_generation_time'):
            print(f"   Generation Time: {self.results['signal_generation_time']:.2f}s")
            print(f"   Signals Generated: {self.results['signals_generated']}")
        else:
            print("   Signal generation tests failed")
        
        # Data Processing
        print("\n📊 DATA PROCESSING:")
        if self.results.get('data_processing_time'):
            print(f"   Processing Time: {self.results['data_processing_time']:.2f}s")
            print(f"   Data Points: {self.results['data_points_processed']:,}")
            rate = self.results['data_points_processed'] / self.results['data_processing_time']
            print(f"   Processing Rate: {rate:,.0f} points/second")
        else:
            print("   Data processing tests failed")
        
        # Overall Performance Grade
        print("\n🎯 PERFORMANCE GRADE:")
        grade = self.calculate_performance_grade()
        print(f"   Overall Grade: {grade}")
        
        print(f"\n⏱️ Total Benchmark Time: {self.results['total_benchmark_time']:.2f}s")
        print("=" * 60)

    def calculate_performance_grade(self):
        """Calculate overall performance grade"""
        score = 0
        max_score = 0
        
        # CPU usage (lower is better)
        if self.results.get('cpu_usage_percent') is not None:
            cpu = self.results['cpu_usage_percent']
            if cpu < 30:
                score += 25
            elif cpu < 50:
                score += 20
            elif cpu < 70:
                score += 15
            else:
                score += 10
            max_score += 25
        
        # Memory usage (lower is better)
        if self.results.get('process_memory_percent') is not None:
            mem = self.results['process_memory_percent']
            if mem < 5:
                score += 25
            elif mem < 10:
                score += 20
            elif mem < 20:
                score += 15
            else:
                score += 10
            max_score += 25
        
        # API response time (lower is better)
        if self.results.get('avg_api_response_time') is not None:
            api = self.results['avg_api_response_time']
            if api < 1:
                score += 25
            elif api < 2:
                score += 20
            elif api < 3:
                score += 15
            else:
                score += 10
            max_score += 25
        
        # Signal generation time (lower is better)
        if self.results.get('signal_generation_time') is not None:
            sig = self.results['signal_generation_time']
            if sig < 1:
                score += 25
            elif sig < 2:
                score += 20
            elif sig < 5:
                score += 15
            else:
                score += 10
            max_score += 25
        
        if max_score > 0:
            percentage = (score / max_score) * 100
            if percentage >= 90:
                return "A+ (Excellent)"
            elif percentage >= 80:
                return "A (Very Good)"
            elif percentage >= 70:
                return "B+ (Good)"
            elif percentage >= 60:
                return "B (Acceptable)"
            else:
                return "C (Needs Improvement)"
        
        return "Unable to calculate"

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run_full_benchmark()
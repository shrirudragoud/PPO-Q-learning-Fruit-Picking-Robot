import os
import time
import json
import numpy as np
from collections import defaultdict

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        """Initialize training logger"""
        self.log_dir = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.step_metrics = defaultdict(dict)
        self.tensorboard_writer = None
        
        # Try to initialize tensorboard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(self.log_dir)
            print(f"Tensorboard logging enabled at {self.log_dir}")
        except:
            print("Tensorboard not available. Using basic logging.")
    
    def log_metrics(self, metrics, step):
        """Log training metrics"""
        # Store metrics
        for name, value in metrics.items():
            self.metrics[name].append(value)
            self.step_metrics[name][step] = value
        
        # Log to tensorboard if available
        if self.tensorboard_writer:
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(name, value, step)
        
        # Save to JSON periodically
        if step % 100 == 0:
            self.save_metrics()
    
    def get_latest_metrics(self):
        """Get the most recent values for all metrics"""
        latest = {}
        for name, values in self.metrics.items():
            if values:
                latest[name] = values[-1]
        return latest
    
    def get_average_metrics(self, window=100):
        """Get moving averages for all metrics"""
        averages = {}
        for name, values in self.metrics.items():
            if values:
                averages[name] = np.mean(values[-window:])
        return averages
    
    def save_metrics(self):
        """Save metrics to disk"""
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.step_metrics, f)
    
    def close(self):
        """Clean up resources"""
        self.save_metrics()
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
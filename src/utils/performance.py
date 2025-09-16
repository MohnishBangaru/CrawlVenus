"""Performance monitoring utilities for DroidBot-GPT framework."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import psutil

from ..core.logger import log


class PerformanceMonitor:
    """Monitor and track system performance metrics."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.max_history_size = 1000
        
    def capture_metrics(self) -> Dict[str, Any]:
        """Capture current system performance metrics.
        
        Returns:
            Dictionary containing current performance metrics.
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024 * 1024)  # MB
            memory_total = memory.total / (1024 * 1024)  # MB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free / (1024 * 1024 * 1024)  # GB
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_bytes_sent = network.bytes_sent
                network_bytes_recv = network.bytes_recv
            except Exception:
                network_bytes_sent = 0
                network_bytes_recv = 0
                
            # Process metrics
            process = psutil.Process()
            process_cpu_percent = process.cpu_percent()
            process_memory_info = process.memory_info()
            process_memory_mb = process_memory_info.rss / (1024 * 1024)  # MB
            
            # Timestamp
            timestamp = time.time()
            
            metrics = {
                'timestamp': timestamp,
                'uptime': timestamp - self.start_time,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'process_percent': process_cpu_percent
                },
                'memory': {
                    'percent': memory_percent,
                    'available_mb': memory_available,
                    'total_mb': memory_total,
                    'process_mb': process_memory_mb
                },
                'disk': {
                    'percent': disk_percent,
                    'free_gb': disk_free
                },
                'network': {
                    'bytes_sent': network_bytes_sent,
                    'bytes_recv': network_bytes_recv
                }
            }
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Limit history size
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
                
            return metrics
            
        except Exception as e:
            log.error(f"Failed to capture performance metrics: {e}")
            return {}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent performance metrics.
        
        Returns:
            Latest performance metrics.
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        else:
            return self.capture_metrics()
    
    def get_metrics_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get a summary of performance metrics over a time window.
        
        Args:
            time_window: Time window in seconds. If None, use all history.
            
        Returns:
            Summary of performance metrics.
        """
        if not self.metrics_history:
            return {}
            
        # Filter metrics by time window
        if time_window is not None:
            current_time = time.time()
            filtered_metrics = [
                m for m in self.metrics_history
                if current_time - m['timestamp'] <= time_window
            ]
        else:
            filtered_metrics = self.metrics_history
            
        if not filtered_metrics:
            return {}
            
        # Calculate averages
        cpu_percents = [m['cpu']['percent'] for m in filtered_metrics]
        memory_percents = [m['memory']['percent'] for m in filtered_metrics]
        process_memory_mbs = [m['memory']['process_mb'] for m in filtered_metrics]
        
        summary = {
            'time_window': time_window,
            'sample_count': len(filtered_metrics),
            'cpu': {
                'average_percent': sum(cpu_percents) / len(cpu_percents),
                'max_percent': max(cpu_percents),
                'min_percent': min(cpu_percents)
            },
            'memory': {
                'average_percent': sum(memory_percents) / len(memory_percents),
                'max_percent': max(memory_percents),
                'min_percent': min(memory_percents),
                'average_process_mb': sum(process_memory_mbs) / len(process_memory_mbs),
                'max_process_mb': max(process_memory_mbs)
            },
            'uptime': time.time() - self.start_time
        }
        
        return summary
    
    def is_performance_acceptable(self, thresholds: Optional[Dict[str, float]] = None) -> bool:
        """Check if current performance is within acceptable thresholds.
        
        Args:
            thresholds: Performance thresholds. If None, use defaults.
            
        Returns:
            True if performance is acceptable, False otherwise.
        """
        if thresholds is None:
            thresholds = {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_percent': 90.0
            }
            
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return False
            
        # Check CPU usage
        if current_metrics['cpu']['percent'] > thresholds.get('cpu_percent', 80.0):
            log.warning(f"High CPU usage: {current_metrics['cpu']['percent']}%")
            return False
            
        # Check memory usage
        if current_metrics['memory']['percent'] > thresholds.get('memory_percent', 85.0):
            log.warning(f"High memory usage: {current_metrics['memory']['percent']}%")
            return False
            
        # Check disk usage
        if current_metrics['disk']['percent'] > thresholds.get('disk_percent', 90.0):
            log.warning(f"High disk usage: {current_metrics['disk']['percent']}%")
            return False
            
        return True
    
    def get_performance_alerts(self) -> List[str]:
        """Get performance alerts based on current metrics.
        
        Returns:
            List of performance alert messages.
        """
        alerts = []
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return ["Unable to capture performance metrics"]
            
        # CPU alerts
        cpu_percent = current_metrics['cpu']['percent']
        if cpu_percent > 90:
            alerts.append(f"Critical CPU usage: {cpu_percent}%")
        elif cpu_percent > 80:
            alerts.append(f"High CPU usage: {cpu_percent}%")
            
        # Memory alerts
        memory_percent = current_metrics['memory']['percent']
        if memory_percent > 95:
            alerts.append(f"Critical memory usage: {memory_percent}%")
        elif memory_percent > 85:
            alerts.append(f"High memory usage: {memory_percent}%")
            
        # Disk alerts
        disk_percent = current_metrics['disk']['percent']
        if disk_percent > 95:
            alerts.append(f"Critical disk usage: {disk_percent}%")
        elif disk_percent > 90:
            alerts.append(f"High disk usage: {disk_percent}%")
            
        # Process memory alerts
        process_memory_mb = current_metrics['memory']['process_mb']
        if process_memory_mb > 1000:  # 1GB
            alerts.append(f"High process memory usage: {process_memory_mb:.1f}MB")
            
        return alerts
    
    def clear_history(self) -> None:
        """Clear the metrics history."""
        self.metrics_history.clear()
        self.start_time = time.time()
        
    def export_metrics(self, filepath: str) -> bool:
        """Export performance metrics to a file.
        
        Args:
            filepath: Path to save the metrics.
            
        Returns:
            True if export successful, False otherwise.
        """
        try:
            from ..utils.file_utils import save_json
            
            export_data = {
                'summary': self.get_metrics_summary(),
                'history': self.metrics_history,
                'alerts': self.get_performance_alerts()
            }
            
            return save_json(export_data, filepath)
            
        except Exception as e:
            log.error(f"Failed to export performance metrics: {e}")
            return False
    
    def get_resource_usage_trend(self, metric: str, time_window: float = 300) -> List[float]:
        """Get trend data for a specific metric over time.
        
        Args:
            metric: Metric to track (e.g., 'cpu.percent', 'memory.percent').
            time_window: Time window in seconds.
            
        Returns:
            List of metric values over time.
        """
        current_time = time.time()
        trend_data = []
        
        for metrics in self.metrics_history:
            if current_time - metrics['timestamp'] <= time_window:
                # Navigate nested dictionary
                value = metrics
                for key in metric.split('.'):
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                        
                if value is not None:
                    trend_data.append(value)
                    
        return trend_data 
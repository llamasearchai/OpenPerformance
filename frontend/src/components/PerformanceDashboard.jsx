import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const PerformanceDashboard = () => {
  const [metrics, setMetrics] = useState([]);

  useEffect(() => {
    const fetchMetrics = async () => {
      const response = await fetch('/api/system/metrics');
      const data = await response.json();
      setMetrics(current => [...current.slice(-10), data]);
    };
    
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard">
      <h2>Real-time Performance Metrics</h2>
      <LineChart width={800} height={400} data={metrics}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="timestamp" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="gpu_utilization" stroke="#8884d8" />
        <Line type="monotone" dataKey="memory_usage" stroke="#82ca9d" />
        <Line type="monotone" dataKey="cpu_usage" stroke="#ffc658" />
      </LineChart>
    </div>
  );
};

export default PerformanceDashboard; 
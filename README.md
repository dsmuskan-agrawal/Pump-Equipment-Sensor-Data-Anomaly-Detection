# Pump Equipment Sensor Data Anomaly Detection
## Running the project

Selectively run needed code blocks from below to general the pkl files and train the model.
```
python Synthetic_Data_Generation.ipynb
```

Now, proceed to execute the anomaly detection app:
```
python Anomaly_Detection_App.py
```

## Summary

This project delivers an **end-to-end anomaly detection framework** for industrial pump systems, combining synthetic sensor simulation, unsupervised machine learning, and natural language insights to enhance operational reliability and enable proactive maintenance.

## Key Components & Achievements

### Realistic Data Synthesis
- Generated **3 months of minute-level sensor data** (temperature, pressure, vibration) for multiple pump types
- Injected anomalies (0.3%) and missing values (0.5%) to mimic real-world conditions
- Paired with handwritten operator logs (every 30 minutes) for correlated event analysis

### Unsupervised Anomaly Detection
- Implemented **K-Means Clustering** (selected over Isolation Forest) for interpretable anomaly detection
- Leveraged timestamp features (hour/day/month) to account for temporal patterns
- Achieved clear separation of anomalies, validated via **80% threshold fuzzy string matching** with manual logs

### Real-Time Insight Generation
- Dynamic warning/anomaly bands (±20% of operational thresholds)
- Severity classification: Normal/Warning/Critical
- **LLM-powered insights** (Mistral model) for contextual query responses

### User-Friendly Application
- **Streamlit dashboard** features:
  - Live sensor monitoring with severity alerts
  - Chatbot for natural language queries
  - Raw data views with anomaly-log correlation proofs

## Business Value
- **Reduced Downtime**: Early detection prevents equipment failure
- **Explainable AI**: Transparent thresholds build operator trust
- **Scalable Architecture**: Ready for Kafka/RabbitMQ integration

## Future Roadmap
- **Real-Time Alerts**: Push notifications via SMS/email
- **Predictive Analytics**: Failure forecasting using historical patterns

*"Transforming sensor data into actionable intelligence for industrial pump systems."*
# 🔧 DevOps Agent

An intelligent infrastructure monitoring and incident response system built with **CrewAI** featuring a Monitor, Root Cause Analyzer, and Incident Responder working together to detect, diagnose, and resolve production issues.

## Features

- 📊 **Real-time system metrics**: CPU, memory, disk, network via psutil
- 📋 **Log analysis**: Error/warning detection with severity classification
- 🔍 **Root cause analysis**: Correlates events to identify primary causes
- 📖 **Runbook execution**: Automated remediation procedures
- 📣 **Incident communication**: Stakeholder update drafts
- 🛡️ **Prevention recommendations**: Post-incident improvements

## Architecture

```
System Metrics + Logs + Alert
          ↓
[Infrastructure Monitor] → System Status Report + Log Analysis
          ↓
[Incident Analyzer] → Root Cause + Impact Assessment
          ↓
[Incident Responder] → Remediation Plan + Communication
```

## Tech Stack

- **Framework**: CrewAI 0.80+
- **LLM**: OpenAI GPT-4o-mini (OpenAI-compatible API)
- **System Monitoring**: psutil (CPU, memory, disk, network)
- **Tools**: SystemMetricsTool, LogAnalysisTool, RunbookTool
- **Pattern**: Sequential multi-agent incident response

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
```

## Usage

```bash
python agent.py
```

Or trigger for a specific incident:
```python
from agent import respond_to_incident

logs = """
2025-01-15 14:03:45 ERROR Failed to connect to Redis
2025-01-15 14:04:15 CRITICAL Cache service down
2025-01-15 14:05:00 ERROR Database query timeout 8500ms
"""

report = respond_to_incident(logs, "Redis cache failure")
print(report)
```

## Runbook Coverage

| Incident Type | Steps | Escalation |
|--------------|-------|------------|
| **High CPU** | Process identification, service restart | PagerDuty if >15min |
| **High Memory** | Memory profiling, cache clearing | OOM monitoring |
| **Disk Full** | Log cleanup, Docker prune, storage provision | Storage team |
| **Service Down** | Status check, restart, failover | On-call engineer |
| **Network Issues** | Connectivity check, DNS, firewall | Network team |

## System Metrics Collected

```
System Metrics [2025-01-15 14:05:00]:
  CPU Usage: 87.3% ⚠️ HIGH
  Memory: 89.1% used (12.4GB / 14GB) ⚠️ HIGH
  Disk: 65.2% used (195GB / 299GB) ✅ Normal
  Network I/O: 1,234.5MB sent / 892.3MB recv
```

## Incident Response Output

1. **Health Assessment**: Current metrics + log summary
2. **Root Cause Analysis**: Primary cause + contributing factors + timeline
3. **Remediation Plan**: Step-by-step recovery actions
4. **Rollback Procedure**: How to undo if fix makes things worse
5. **Stakeholder Update**: Ready-to-send communication draft
6. **Prevention Measures**: Monitoring alerts + process improvements

## Integration

```python
# Integrate with monitoring systems
from agent import respond_to_incident

# Called by your alerting system (Datadog, PagerDuty, etc.)
def handle_alert(alert_type: str, logs: str):
    report = respond_to_incident(logs, alert_type)
    send_to_slack("#incidents", report)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |

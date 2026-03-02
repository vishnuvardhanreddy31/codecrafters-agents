"""
DevOps Agent
Uses CrewAI with specialized agents (Monitor, Analyzer, Responder)
to detect infrastructure issues, analyze root causes, and coordinate incident response.
"""

import os
import psutil
import random
from datetime import datetime, timedelta
from typing import List, Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# ── Tools ──────────────────────────────────────────────────────────────────────
class SystemMetricsInput(BaseModel):
    detailed: bool = Field(default=False, description="Whether to return detailed metrics")


class SystemMetricsTool(BaseTool):
    name: str = "get_system_metrics"
    description: str = "Collect current system performance metrics: CPU, memory, disk, and network."
    args_schema: type[BaseModel] = SystemMetricsInput

    def _run(self, detailed: bool = False) -> str:
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net = psutil.net_io_counters()

            result = (
                f"System Metrics [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:\n"
                f"  CPU Usage: {cpu:.1f}%"
                + (" ⚠️ HIGH" if cpu > 80 else " ✅ Normal")
                + f"\n  Memory: {mem.percent:.1f}% used ({mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB)"
                + (" ⚠️ HIGH" if mem.percent > 85 else " ✅ Normal")
                + f"\n  Disk: {disk.percent:.1f}% used ({disk.used / 1e9:.1f}GB / {disk.total / 1e9:.1f}GB)"
                + (" ⚠️ CRITICAL" if disk.percent > 90 else " ✅ Normal")
                + f"\n  Network I/O: {net.bytes_sent / 1e6:.1f}MB sent / {net.bytes_recv / 1e6:.1f}MB recv"
            )

            if detailed:
                result += f"\n  CPU Count: {psutil.cpu_count()}"
                result += f"\n  Boot time: {datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M')}"
                try:
                    load_avg = os.getloadavg()
                    result += f"\n  Load avg (1/5/15min): {load_avg[0]:.2f} / {load_avg[1]:.2f} / {load_avg[2]:.2f}"
                except AttributeError:
                    pass  # Windows doesn't support getloadavg

            return result
        except Exception as e:
            return f"Error collecting metrics: {e}"


class LogAnalysisInput(BaseModel):
    log_content: str = Field(description="Log content to analyze")
    severity_filter: str = Field(default="all", description="Filter: all | error | warning | critical")


class LogAnalysisTool(BaseTool):
    name: str = "analyze_logs"
    description: str = "Analyze application logs for errors, warnings, and anomalies."
    args_schema: type[BaseModel] = LogAnalysisInput

    def _run(self, log_content: str, severity_filter: str = "all") -> str:
        lines = log_content.strip().split("\n")
        errors, warnings, criticals, info = [], [], [], []

        for line in lines:
            line_lower = line.lower()
            if "critical" in line_lower or "fatal" in line_lower:
                criticals.append(line.strip())
            elif "error" in line_lower or "exception" in line_lower or "failed" in line_lower:
                errors.append(line.strip())
            elif "warn" in line_lower or "warning" in line_lower:
                warnings.append(line.strip())
            else:
                info.append(line.strip())

        result = f"Log Analysis ({len(lines)} lines):\n"
        result += f"  🔴 Critical: {len(criticals)}\n"
        result += f"  🟠 Errors: {len(errors)}\n"
        result += f"  🟡 Warnings: {len(warnings)}\n"
        result += f"  ℹ️  Info: {len(info)}\n\n"

        if criticals and severity_filter in ("all", "critical"):
            result += "Critical Issues:\n" + "\n".join(f"  {c}" for c in criticals[:5]) + "\n"
        if errors and severity_filter in ("all", "error"):
            result += "Recent Errors:\n" + "\n".join(f"  {e}" for e in errors[:5]) + "\n"
        if warnings and severity_filter in ("all", "warning"):
            result += "Warnings:\n" + "\n".join(f"  {w}" for w in warnings[:3]) + "\n"

        return result


class RunbookInput(BaseModel):
    incident_type: str = Field(description="Type of incident to get runbook for")


class RunbookTool(BaseTool):
    name: str = "get_runbook"
    description: str = "Retrieve incident runbook procedures for common infrastructure issues."
    args_schema: type[BaseModel] = RunbookInput

    def _run(self, incident_type: str) -> str:
        runbooks = {
            "high_cpu": (
                "High CPU Runbook:\n"
                "1. Identify top processes: ps aux --sort=-%cpu | head -10\n"
                "2. Check for runaway processes or infinite loops\n"
                "3. Review application logs for errors during spike\n"
                "4. Check for scheduled tasks that may be running\n"
                "5. If >95% for >10min: restart offending service\n"
                "6. Scale horizontally if load is legitimate\n"
                "7. Alert: PagerDuty if not resolved in 15 minutes"
            ),
            "high_memory": (
                "High Memory Runbook:\n"
                "1. Identify memory-heavy processes: ps aux --sort=-%mem | head -10\n"
                "2. Check for memory leaks in application logs\n"
                "3. Review recent deployments for memory regressions\n"
                "4. Clear application caches if safe to do so\n"
                "5. If >95%: restart low-priority services first\n"
                "6. Escalate if OOM killer starts: check /var/log/kern.log\n"
                "7. Schedule post-incident: profiling session"
            ),
            "disk_full": (
                "Disk Full Runbook:\n"
                "1. Identify large files: du -sh /* | sort -hr | head -20\n"
                "2. Clear old log files: find /var/log -name '*.gz' -delete\n"
                "3. Clear Docker images: docker system prune -f\n"
                "4. Rotate application logs immediately\n"
                "5. Alert DevOps team to provision additional storage\n"
                "6. Set up disk space monitoring alerts (80% warning, 90% critical)\n"
                "7. Review log retention policies"
            ),
            "service_down": (
                "Service Down Runbook:\n"
                "1. Check service status: systemctl status <service>\n"
                "2. Review service logs: journalctl -u <service> -n 100\n"
                "3. Check port availability: netstat -tlnp | grep <port>\n"
                "4. Attempt restart: systemctl restart <service>\n"
                "5. If restart fails: check dependencies and config files\n"
                "6. Failover to standby if available\n"
                "7. Notify on-call team via PagerDuty\n"
                "8. Create incident ticket immediately"
            ),
            "network": (
                "Network Issue Runbook:\n"
                "1. Check connectivity: ping gateway, external hosts\n"
                "2. Review network interface: ip addr show\n"
                "3. Check routing table: ip route show\n"
                "4. Verify DNS: nslookup, dig\n"
                "5. Check firewall rules: iptables -L\n"
                "6. Review load balancer health checks\n"
                "7. Coordinate with network team for hardware issues"
            ),
        }
        incident_lower = incident_type.lower().replace(" ", "_")
        for key, runbook in runbooks.items():
            if key in incident_lower or incident_lower in key:
                return f"Runbook for '{incident_type}':\n\n{runbook}"

        return (
            f"No specific runbook for '{incident_type}'.\n\n"
            "General Incident Response:\n"
            "1. Assess impact and severity\n"
            "2. Notify affected teams\n"
            "3. Begin root cause analysis\n"
            "4. Implement temporary fix if possible\n"
            "5. Document all actions taken\n"
            "6. Schedule post-mortem review"
        )


system_metrics_tool = SystemMetricsTool()
log_analysis_tool = LogAnalysisTool()
runbook_tool = RunbookTool()

# ── Agents ─────────────────────────────────────────────────────────────────────
infrastructure_monitor = Agent(
    role="Infrastructure Monitor",
    goal="Continuously monitor system health and detect performance anomalies.",
    backstory=(
        "You are an experienced SRE (Site Reliability Engineer) who monitors "
        "complex distributed systems. You have a keen eye for spotting unusual "
        "patterns and correlating metrics to identify issues before they escalate."
    ),
    tools=[system_metrics_tool, log_analysis_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

incident_analyzer = Agent(
    role="Incident Root Cause Analyzer",
    goal="Analyze incidents to determine root cause and assess impact.",
    backstory=(
        "You are a systems architect with deep expertise in debugging complex "
        "infrastructure failures. You excel at identifying root causes from "
        "limited data and correlating events across systems."
    ),
    tools=[log_analysis_tool, runbook_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

incident_responder = Agent(
    role="Incident Response Coordinator",
    goal="Coordinate the incident response and create actionable remediation plans.",
    backstory=(
        "You are an incident commander who has managed hundreds of production "
        "outages. You know how to prioritize actions, communicate clearly, and "
        "restore services quickly while minimizing business impact."
    ),
    tools=[runbook_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ── Sample logs for demo ───────────────────────────────────────────────────────
SAMPLE_LOGS = """
2025-01-15 14:00:01 INFO Application started successfully
2025-01-15 14:01:15 INFO Request processed: GET /api/users (200) 45ms
2025-01-15 14:02:30 WARNING Database connection pool at 85% capacity
2025-01-15 14:03:45 ERROR Failed to connect to Redis: Connection refused (attempt 1/3)
2025-01-15 14:04:00 ERROR Failed to connect to Redis: Connection refused (attempt 2/3)
2025-01-15 14:04:15 CRITICAL Redis connection failed after 3 attempts - cache disabled
2025-01-15 14:04:30 ERROR Uncached database query taking 8500ms - potential timeout
2025-01-15 14:04:45 WARNING High memory usage: 89% (heap: 12.4GB / 14GB)
2025-01-15 14:05:00 ERROR Request timeout: GET /api/products (504) 30000ms
2025-01-15 14:05:15 ERROR NullPointerException in ProductService.getRecommendations()
2025-01-15 14:05:30 CRITICAL Service health check FAILED - 3 consecutive failures
2025-01-15 14:05:45 ERROR Database connection timeout after 5000ms
"""


def build_devops_crew(logs: str, incident_type: str = "general") -> Crew:
    monitoring_task = Task(
        description=(
            f"Perform system health assessment:\n\n"
            f"1. Use get_system_metrics to collect current performance data\n"
            f"2. Use analyze_logs with these application logs:\n{logs}\n\n"
            "Provide: current system status, key metrics summary, and top concerns."
        ),
        agent=infrastructure_monitor,
        expected_output=(
            "System health report with: current metrics, log analysis summary, "
            "identified anomalies, and severity assessment."
        ),
    )

    analysis_task = Task(
        description=(
            f"Analyze the incident and determine root cause:\n"
            f"Incident type: {incident_type}\n\n"
            "1. Review monitoring findings\n"
            "2. Use analyze_logs to identify error patterns and timeline\n"
            "3. Use get_runbook to understand the incident type\n"
            "4. Determine: root cause, contributing factors, blast radius\n"
            "5. Assess business impact and urgency"
        ),
        agent=incident_analyzer,
        expected_output=(
            "Root cause analysis with: primary cause, contributing factors, "
            "timeline of events, impact assessment, and initial diagnosis."
        ),
        context=[monitoring_task],
    )

    response_task = Task(
        description=(
            "Create and execute incident response plan:\n\n"
            "1. Get the appropriate runbook using get_runbook\n"
            "2. Prioritize immediate actions (stop the bleeding)\n"
            "3. Create step-by-step remediation plan\n"
            "4. Define rollback procedure if remediation fails\n"
            "5. Write stakeholder communication draft\n"
            "6. Schedule post-incident review items\n"
            "7. Define monitoring alerts to prevent recurrence"
        ),
        agent=incident_responder,
        expected_output=(
            "Complete incident response plan with: immediate actions, "
            "remediation steps, rollback plan, stakeholder update, and prevention measures."
        ),
        context=[monitoring_task, analysis_task],
    )

    return Crew(
        agents=[infrastructure_monitor, incident_analyzer, incident_responder],
        tasks=[monitoring_task, analysis_task, response_task],
        process=Process.sequential,
        verbose=True,
    )


def respond_to_incident(logs: str, incident_type: str = "service degradation") -> str:
    crew = build_devops_crew(logs, incident_type)
    result = crew.kickoff()
    return str(result)


def main():
    print("🔧 DevOps Agent - Incident Response System\n" + "=" * 60)
    print("Simulating production incident with Redis failure and high memory...\n")
    result = respond_to_incident(SAMPLE_LOGS, "Redis cache failure with memory pressure")
    print("\n" + "=" * 60)
    print("📋 Incident Response Report:")
    print("=" * 60)
    print(result)
    print("\n✅ Incident response complete!")


if __name__ == "__main__":
    main()

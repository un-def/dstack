import json
import re
import time
import urllib
from collections import defaultdict
from typing import Optional, Dict, List, Generator

from botocore.client import BaseClient

from dstack.aws import jobs, runs
from dstack.backend import LogEvent, LogEventSource
from dstack.jobs import JobApp

POLL_LOGS_RATE_SECS = 1


def _render_log_message(s3_client: BaseClient, bucket_name: str, log_message: str,
                        repo_user_name: str, repo_name: str,
                        job_id: Optional[str],
                        job_host_names: Dict[str, Optional[str]],
                        job_ports: Dict[str, Optional[List[int]]],
                        job_apps: Dict[str, Optional[List[JobApp]]]) -> str:
    if job_id and job_id not in job_host_names:
        job = jobs.get_job(s3_client, bucket_name, repo_user_name, repo_name, job_id)
        job_host_names[job_id] = job.host_name or "none"
        job_ports[job_id] = job.ports
        job_apps[job_id] = job.apps
    message = json.loads(log_message.strip())["log"]
    host_name = job_host_names[job_id]
    ports = job_ports[job_id]
    apps = job_apps[job_id]
    pat = re.compile(f'http://(localhost|0.0.0.0|{host_name}):[\\S]*[^(.+)\\s\\n\\r]')
    if re.search(pat, message):
        if host_name != "none" and ports and apps:
            for app in apps:
                port = ports[app["port_index"]]
                url_path = app.get("url_path") or ""
                url_query_params = app.get("url_query_params")
                url_query = ("?" + urllib.parse.urlencode(url_query_params)) if url_query_params else ""
                app_url = f"http://{host_name}:{port}"
                if url_path or url_query_params:
                    app_url += "/"
                    if url_query_params:
                        app_url += url_query
                message = re.sub(pat, app_url, message)
    return message


def _get_latest_events_and_timestamp(event_ids_per_timestamp):
    if event_ids_per_timestamp:
        newest_timestamp = max(event_ids_per_timestamp.keys())
        event_ids_per_timestamp = defaultdict(
            set, {newest_timestamp: event_ids_per_timestamp[newest_timestamp]}
        )
    return event_ids_per_timestamp


def _reset_filter_log_events_params(fle_kwargs, event_ids_per_timestamp):
    if event_ids_per_timestamp:
        fle_kwargs['startTime'] = max(
            event_ids_per_timestamp.keys()
        )
    fle_kwargs.pop('nextToken', None)


def _do_filter_log_events(s3_client: BaseClient, logs_client: BaseClient, bucket_name: str,
                          repo_user_name: str, repo_name: str, run_name: str,
                          filter_logs_events_kwargs: dict):
    event_ids_per_timestamp = defaultdict(set)
    counter = 0
    while True:
        response = logs_client.filter_log_events(**filter_logs_events_kwargs)

        for event in response["events"]:
            if event["eventId"] not in event_ids_per_timestamp[event["timestamp"]]:
                event_ids_per_timestamp[event["timestamp"]].add(event["eventId"])
                yield event
        event_ids_per_timestamp = _get_latest_events_and_timestamp(
            event_ids_per_timestamp
        )
        if "nextToken" in response:
            filter_logs_events_kwargs["nextToken"] = response["nextToken"]
        else:
            _reset_filter_log_events_params(
                filter_logs_events_kwargs,
                event_ids_per_timestamp
            )
            time.sleep(POLL_LOGS_RATE_SECS)
            counter = counter + 1
            if counter % 3 == 0:
                run = runs.get_runs(s3_client, bucket_name, repo_user_name, repo_name, run_name)[0]
                if run.status.is_finished():
                    break


def create_log_group_if_not_exists(logs_client: BaseClient, bucket_name: str, log_group_name: str):
    response = logs_client.describe_log_groups(logGroupNamePrefix=log_group_name)
    if not response["logGroups"] or not any(
            filter(lambda g: g["logGroupName"] == log_group_name, response["logGroups"])):
        logs_client.create_log_group(
            logGroupName=log_group_name,
            tags={
                'owner': 'dstack',
                'dstack_bucket': bucket_name,
            }
        )


def create_log_stream(logs_client: BaseClient, log_group_name: str, run_name: str):
    logs_client.create_log_stream(logGroupName=log_group_name, logStreamName=run_name)


def poll_logs(s3_client: BaseClient, logs_client: BaseClient, bucket_name: str,
              repo_user_name: str, repo_name: str,
              run_name: str, start_time: int, attached: bool) -> Generator[LogEvent, None, None]:
    filter_logs_events_kwargs = {
        "logGroupName": f"/dstack/jobs/{bucket_name}/{repo_user_name}/{repo_name}",
        "logStreamNames": [run_name],
        "startTime": start_time,
        "interleaved": True,
    }
    job_host_names = {}
    job_ports = {}
    job_apps = {}

    try:
        if attached:
            for event in _do_filter_log_events(s3_client, logs_client, bucket_name, repo_user_name, repo_name, run_name,
                                               filter_logs_events_kwargs):
                log_message = _render_log_message(s3_client, bucket_name, event["message"], repo_user_name, repo_name,
                                                  event.get("job_id"), job_host_names, job_ports, job_apps)
                yield LogEvent(event["timestamp"], event.get("job_id"), log_message,
                               LogEventSource.STDOUT if event["source"] == "stdout" else LogEventSource.STDERR)
        else:
            paginator = logs_client.get_paginator("filter_log_events")
            for page in paginator.paginate(**filter_logs_events_kwargs):
                for event in page["events"]:
                    log_message = _render_log_message(s3_client, bucket_name, event["message"], repo_user_name,
                                                      repo_name, event.get("job_id"), job_host_names, job_ports,
                                                      job_apps)
                    yield LogEvent(event["timestamp"], event.get("job_id"), log_message,
                                   LogEventSource.STDOUT if event["source"] == "stdout" else LogEventSource.STDERR)
    except Exception as e:
        if hasattr(e, "response") and e.response.get("Error") and e.response["Error"].get(
                "Code") == "ResourceNotFoundException":
            return
        else:
            raise e

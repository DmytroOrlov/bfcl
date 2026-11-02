#!/usr/bin/env python3
"""Local, dependency-free BFCL HTTP preflight / optional effort-injection proxy.

Optional BFCL_CAPTURE_TOOL_FEEDBACK=1 saves the last tool result in the
trace and prints it to stderr. Tool outputs can contain sensitive data;
keep the trace local and inspect before sharing. No requests are changed
by this option.

Run: BFCL_UPSTREAM=http://127.0.0.1:8002 python bfcl_reasoning_proxy.py
Point EvalScope API_URL to http://127.0.0.1:18002/v1.
Logs offered tool names and response metadata, but not prompts, arguments, or credentials.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from threading import Lock

UPSTREAM = os.environ.get('BFCL_UPSTREAM', 'http://127.0.0.1:8002').rstrip('/')
UPSTREAM_MODEL = os.environ.get('BFCL_UPSTREAM_MODEL')
PORT = int(os.environ.get('BFCL_PROXY_PORT', '18002'))
EXPECTED = os.environ.get('BFCL_EXPECT_EFFORT', 'xhigh')
INJECT = os.environ.get('BFCL_INJECT_EFFORT') == '1'
COUNT = [0]
TRACE_LOCK = Lock()
FORCE_GRAMMAR = os.environ.get('BFCL_FORCE_GRAMMAR') == '1'
TRACE_FILE = os.environ.get('BFCL_TRACE_FILE', '/tmp/bfcl_proxy_trace.jsonl')
CAPTURE_TOOL_FEEDBACK = os.environ.get('BFCL_CAPTURE_TOOL_FEEDBACK') == '1'
TOOL_FEEDBACK_MAX_CHARS = int(os.environ.get('BFCL_TOOL_FEEDBACK_MAX_CHARS', '1200'))

HOP_HEADERS = {'host', 'connection', 'transfer-encoding', 'content-length', 'keep-alive'}


class Proxy(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        trace = None
        size = int(self.headers.get('Content-Length', '0'))
        data = self.rfile.read(size) if size else None
        if self.command == 'POST' and self.path.split('?', 1)[0].rstrip('/').endswith('/chat/completions'):
            try:
                payload = json.loads(data)
            except (ValueError, TypeError):
                return self.reply(400, b'Invalid JSON request')
            effort = payload.get('reasoning_effort')
            with TRACE_LOCK:
                COUNT[0] += 1
                seq = COUNT[0]
            offered = [t.get('function', {}).get('name') for t in payload.get('tools', [])
                       if isinstance(t, dict) and isinstance(t.get('function'), dict)]
            trace = {
                'request': seq,
                'model': payload.get('model'),
                'tool_choice': payload.get('tool_choice', '(default)'),
                'parallel_before': payload.get('parallel_tool_calls', '(default)'),
                'offered_tools': offered,
                'reasoning_effort_before': effort,
                'message_count': len(payload.get('messages') or []),
                'roles_tail': [msg.get('role') for msg in (payload.get('messages') or [])[-8:]
                               if isinstance(msg, dict)],
                'tool_result_count': sum(msg.get('role') == 'tool' for msg in (payload.get('messages') or [])
                                         if isinstance(msg, dict)),
            }
            if CAPTURE_TOOL_FEEDBACK:
                messages = payload.get('messages') or []
                last = messages[-1] if messages else None
                if isinstance(last, dict) and last.get('role') == 'tool':
                    value = last.get('content', '')
                    if not isinstance(value, str):
                        value = json.dumps(value, ensure_ascii=False, default=str)
                    tool_feedback = {
                        'tool_call_id': last.get('tool_call_id'),
                        'tool_name': last.get('name'),
                        'content': value[:TOOL_FEEDBACK_MAX_CHARS],
                        'content_length': len(value),
                        'truncated': len(value) > TOOL_FEEDBACK_MAX_CHARS,
                    }
                    trace['last_tool_feedback'] = tool_feedback
                    print(f'[BFCL TOOL] #{seq} result_of={tool_feedback["tool_call_id"]} '
                          f'name={tool_feedback["tool_name"]!r} '
                          f'content={tool_feedback["content"]!r}',
                          file=sys.stderr, flush=True)
            if FORCE_GRAMMAR and offered:
                payload['parallel_tool_calls'] = False
            trace['parallel_sent'] = payload.get('parallel_tool_calls', '(default)')
            if effort is None and INJECT:
                payload['reasoning_effort'] = EXPECTED
                data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                action = 'INJECTED'
            elif effort == EXPECTED:
                action = 'PASSED'
            else:
                print(f"[BFCL HTTP] #{COUNT[0]} REJECTED effort={effort!r}; "
                      f"expected={EXPECTED!r}; model={payload.get('model')!r}",
                      file=sys.stderr, flush=True)
                return self.reply(422, b'BFCL DEBUG: reasoning_effort mismatch or missing')
            trace['effort_action'] = action
            trace['reasoning_effort_sent'] = payload.get('reasoning_effort')
            if COUNT[0] <= 3 or COUNT[0] % 100 == 0:
                print(f"[BFCL HTTP] #{seq} {action} effort={EXPECTED!r}; "
                      f"parallel={trace['parallel_sent']!r}; offered={len(offered)}",
                      file=sys.stderr, flush=True)
            if UPSTREAM_MODEL:
                trace['upstream_model'] = UPSTREAM_MODEL
                payload['model'] = UPSTREAM_MODEL
            # Re-serialize after possible parallel_tool_calls modification.
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')

        headers = {key: value for key, value in self.headers.items()
                   if key.lower() not in HOP_HEADERS}
        try:
            req = Request(UPSTREAM + self.path, data=data, headers=headers,
                          method=self.command)
            try:
                with urlopen(req, timeout=1300) as upstream:
                    body = upstream.read()
                    if trace is not None:
                        self.trace_response(trace, upstream.status, body)
                    return self.respond(upstream.status, upstream.headers, body)
            except HTTPError as error:
                body = error.read()
                if trace is not None:
                    self.trace_response(trace, error.code, body)
                return self.respond(error.code, error.headers, body)
        except URLError as error:
            if trace is not None:
                self.trace_response(trace, 502, str(error).encode()[:1000])
            return self.reply(502, str(error).encode()[:1000])

    def trace_response(self, trace, status, body):
        trace['status'] = status
        try:
            reply = json.loads(body)
            choice = (reply.get('choices') or [{}])[0]
            message = choice.get('message') or {}
            calls = message.get('tool_calls') or []
            offered = set(trace['offered_tools'])
            trace['finish_reason'] = choice.get('finish_reason')
            trace['tool_calls'] = [
                {
                    'name': (call.get('function') or {}).get('name'),
                    'offered': (call.get('function') or {}).get('name') in offered,
                    'arguments_json_ok': is_json((call.get('function') or {}).get('arguments')),
                }
                for call in calls if isinstance(call, dict)
            ]
            content = message.get('content') or ''
            trace['content_chars'] = len(content) if isinstance(content, str) else None
            trace['xml_tool_tag_in_content'] = '<tool_call>' in content if isinstance(content, str) else None
            trace['error'] = reply.get('error')
        except (ValueError, TypeError, KeyError) as e:
            trace['response_parse_error'] = str(e)
        with TRACE_LOCK:
            with open(TRACE_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace, ensure_ascii=False, default=str) + '\n')
        if status != 200 or trace['request'] <= 3 or trace['request'] % 25 == 0 or any(
            not c['offered'] or not c['arguments_json_ok'] for c in trace.get('tool_calls', [])
        ) or trace.get('xml_tool_tag_in_content'):
            print(f"[BFCL TRACE] #{trace['request']} HTTP {status} "
                  f"finish={trace.get('finish_reason')} tools={trace.get('tool_calls', [])} "
                  f"xml={trace.get('xml_tool_tag_in_content')} ", file=sys.stderr, flush=True)

    def reply(self, status, data):
        self.respond(status, {}, data)

    def respond(self, status, headers, data):
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() not in HOP_HEADERS and key.lower() not in {'date', 'server'}:
                self.send_header(key, value)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)


def is_json(s):
    if not isinstance(s, str):
        return False
    try:
        json.loads(s)
        return True
    except (ValueError, TypeError):
        return False


if __name__ == '__main__':
    print(f'[BFCL HTTP] proxy 127.0.0.1:{PORT} -> {UPSTREAM}; expect {EXPECTED!r}; inject={INJECT}; force_grammar={FORCE_GRAMMAR}; trace={TRACE_FILE}',
          file=sys.stderr, flush=True)
    try:
        ThreadingHTTPServer(('127.0.0.1', PORT), Proxy).serve_forever()
    except KeyboardInterrupt:
        print(f'\n[BFCL HTTP] checked requests: {COUNT[0]}', file=sys.stderr)

"""CPU regressions for the first real Krea 2 memory queue's integration failures.

Exercise the actual source fragments without importing the model or starting the web server.
"""
import ast
import io
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]

def _tree(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))

def _exec_nodes(nodes, env):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<integration fragment>", "exec"), env)

def test_parameter_names_do_not_replace_dataset_group():
    tree = _tree("src/fizgig/families/train.py")
    node, = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "optimizer.__class__.__name__ == 'CompressedMuonME'"]
    # Execute the actual name mapping and loop, omitting only the import.
    factor = object()
    adapter = SimpleNamespace(named_parameters=lambda: [("0.weight", factor)])
    net = SimpleNamespace(wrapped={"block": SimpleNamespace(adapters={"train": adapter})})
    optimizer = SimpleNamespace(param_groups=[{"params": [factor]}])
    dataset = SimpleNamespace(datasets=[])
    env = dict(net=net, optimizer=optimizer, group=dataset, TRAINABLE="train")
    _exec_nodes([n for n in node.body if not isinstance(n, ast.ImportFrom)], env)
    assert env["group"] is dataset
    assert optimizer.param_groups[0]["param_names"] == ["block.0.weight"]

def test_both_forward_paths_have_phase_scope():
    tree = _tree("src/fizgig/families/train.py")
    node, = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "_many" and any(isinstance(x, ast.Call) and ast.unparse(x.func) == "_batch_loss" for x in ast.walk(n))]
    for nodes, callee in ((node.body, "_batch_loss"), (node.orelse, "driver.training_loss")):
        scope, = [x for x in nodes if isinstance(x, ast.With) and any(isinstance(y, ast.Call) and ast.unparse(y.func) == callee for y in ast.walk(x))]
        assert ast.unparse(scope.items[0].context_expr) == "_memory_phase('forward', global_step + 1)"

def test_disk_logger_retains_research_records_on_progress_redraw():
    tree = _tree("src/fizgig/web/server.py")
    fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "append_log"]
    lock = threading.RLock()
    stream = io.StringIO()
    env = dict(_lock=lock, _log_n=[0], _log=[], _log_event=threading.Condition(lock),
               _run_log={"file": stream, "bar_at": time.time()}, LOG_CLEAR="clear",
               _BAR=re.compile(r"\d+%\|"), time=time)
    _exec_nodes([fn], env)
    append = env["append_log"]
    append("steps: 15%|ordinary progress\n")
    assert not stream.getvalue()
    for marker in ("memory", "memory-state", "memory-config", "memory-loss", "fit"):
        line = f"steps: 15%|INFO:[geometry-{marker}] test\n"
        append(line)
        assert line in stream.getvalue()

def test_loss_parser_accepts_progress_prefix():
    from fizgig.web.memory_benchmark import parse_memory_log
    out = parse_memory_log('steps: 15%|INFO:[geometry-memory-loss] {"step":2,"loss":0.123,"items":1}\n')
    assert out["memory_losses"] == [{"step":2,"loss":0.123,"items":1}]

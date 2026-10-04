"""UNEXECUTED opt-in execution provider; original critic/optimizer is external.

This provider cannot initialize factors, fit a head, change a cotangent or step
an optimizer. Proposed core hooks preserve the original resident route when the
provider is None. Root must qualify those hooks on a fresh integration fixture.
"""
import hashlib
import json
import resource
import time
from pathlib import Path

import torch

from src import kernel_mean_ce as core
from src import kernel_mean_row_tile as rowtile
from src.io import array_digest
from src.kernel_mean_row_tile import FrozenPhiRows, TiledKernelMeanMoments, bounded_outer_gradient
from src.shared_features import _tensor_identity

SCHEMA = 1
EXECUTION_MODE = "original_Phi_row_tile_kernel_mean_uniform_CE_execution_v1"
EXECUTION_KEYS = {"schema", "mode", "provider_source", "operator_source", "rows",
    "arithmetic_acceptance", "bounds"}
BOUNDS = dict(seconds=300, allocated_bytes=4*1024**3, reserved_bytes=6*1024**3,
    RSS_bytes=16*1024**3, tile_payload_bytes=512*1024**2)


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha_ref(ref):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
        and type(ref["path"]) is str and Path(ref["path"]).is_absolute()
        and type(ref["sha256"]) is str and len(ref["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in ref["sha256"]), "Typed immutable execution source reference required")
    return ref


def execution_context(context):
    """Additional metadata only; original eight-field critic context persists."""
    refs = context.get("source_refs", {}) if isinstance(context, dict) else {}
    value = refs.get("execution_backend")
    _require(isinstance(value, dict) and set(value) == EXECUTION_KEYS
        and type(value["schema"]) is int and value["schema"] == SCHEMA
        and value["mode"] == EXECUTION_MODE and value["bounds"] == BOUNDS
        and isinstance(value["bounds"], dict)
        and all(type(v) is int for v in value["bounds"].values()),
        "Distinct frozen row-tile execution mode/context missing")
    for key in ("provider_source", "operator_source", "arithmetic_acceptance"):
        _sha_ref(value[key])
    rows = value["rows"]
    _require(isinstance(rows, dict) and type(rows.get("schema")) is int and rows["schema"] == 1
        and rows.get("backend") == "original_Phi_FP64_complete_row_tile_native_NODE_v1"
        and rows.get("shape") == [refs.get("nodes"), refs.get("basis")]
        and rows.get("dtype") == "float64" and rows.get("tile_payload_limit") == BOUNDS["tile_payload_bytes"]
        and rows.get("content_digest") == context.get("asset_descriptors", {}).get("Phi_identity", {}).get("phi_digest"),
        "Original cached Phi/source execution descriptor differs")
    _json(value)
    return value


def expected_config(options_without_save_resume, context):
    """Pure JSON extension; no live provider object is serialized."""
    execution = execution_context(context)
    return dict(options_without_save_resume, data_digest=context["source_refs"]["data_digest"],
        kernel_mean_mode=core.MODE, kernel_mean_context=context,
        kernel_mean_execution=execution)


def _summary(events):
    totals = {}
    for event in events:
        for key, value in event["evidence"].get("counts", {}).items():
            _require(type(value) is int and value >= 0, "Untyped execution counter")
            totals[key] = totals.get(key, 0) + value
    return totals


def validate_execution_state(state, context, step, work=None):
    """Metadata link for record/history/current/retained frontier/resume guards."""
    execution = execution_context(context)
    _require(type(step) is int and step >= 0 and isinstance(state, dict)
        and set(state) == {"schema", "execution", "step", "events", "summary", "state_digest"}
        and type(state["schema"]) is int and state["schema"] == SCHEMA
        and type(state["step"]) is int and state["step"] == step
        and core._seal(state["execution"]) == core._seal(execution),
        "Execution state/schema/source step changed")
    events = state["events"]
    _require(isinstance(events, list) and len(events) >= 2*(step+1) and len(events)%2 == 0,
        "Complete critic/outer event prefix missing")
    prior_step = -1
    for index, event in enumerate(events):
        _require(isinstance(event, dict) and set(event) == {"index", "step", "kind", "evidence"}
            and type(event["index"]) is int and event["index"] == index and type(event["step"]) is int
            and prior_step <= event["step"] <= step
            and event["kind"] == ("critic_moments" if index%2 == 0 else "source_outer"),
            "Execution event order or source step changed")
        if index%2:
            _require(event["step"] == events[index-1]["step"], "Critic/head/outer endpoints differ")
        prior_step = event["step"]
        ev = event["evidence"]
        _require(isinstance(ev, dict) and not ev.get("failures")
            and "partial_arrays" not in ev and ev.get("partial_result_is_complete", True) is True,
            "Failed/incomplete operator evidence cannot be an accepted prefix")
        counter = ev.get("counts", {})
        if index%2 == 0:
            _require(counter.get("moment_forward_attempts") == counter.get("moment_forward_completed") == 1
                and counter.get("moment_backward_attempts", 0) == counter.get("moment_backward_completed", 0)
                and counter.get("moment_backward_completed", 0) in (0,1), "Critic execution lost completed counts")
        else:
            _require(counter.get("outer_CE_attempts") == counter.get("outer_CE_completed") == 1,
                "Source outer execution lost completed counts")
    totals = _summary(events)
    _require(state["summary"] == totals
        and totals.get("moment_backward_completed", 0) == step
        and totals.get("moment_forward_completed", 0) == totals.get("outer_CE_completed", 0)
        and state["state_digest"] == core._seal({k:v for k,v in state.items() if k != "state_digest"}),
        "Execution aggregate/endpoint/seal changed")
    if work is not None:
        _require(totals.get("moment_forward_completed",0) == work["moment_forward_calls"]
            and totals.get("moment_backward_completed",0) == work["moment_backward_calls"]
            and totals.get("outer_CE_completed",0) == work["head_interfaces"]
            and work["P_updates"] == step, "Execution telemetry differs from actual core work")
    return state


class RowTileExecution:
    def __init__(self, rows, context, stop=lambda: False):
        self.execution = _json(execution_context(context))
        _require(type(rows) is FrozenPhiRows and rows.descriptor() == self.execution["rows"]
            and callable(stop), "Typed frozen row source/provider required")
        self.rows, self.context, self.external_stop = rows, _json(context), stop
        self.started = time.monotonic()
        self.events, self.source_validation = [], {}
        self.admitted = False

    def stop(self):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            _require(torch.cuda.max_memory_allocated() <= BOUNDS["allocated_bytes"]
                and torch.cuda.max_memory_reserved() <= BOUNDS["reserved_bytes"], "Execution GPU peak bound exceeded")
        _require(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024 <= BOUNDS["RSS_bytes"], "Execution RSS bound exceeded")
        return self.external_stop() or time.monotonic()-self.started > BOUNDS["seconds"]

    def check(self):
        if self.stop():
            raise InterruptedError("Bounded row-tile critic invocation stopped; retain accepted prefix")

    def admit(self, z, q, assignment, initial, options, context):
        """Same native admission, replacing only resident-specific Phi checks."""
        self.check()
        _require(core._seal(context) == core._seal(self.context), "Provider/caller context mismatch")
        refs, assets, basis = core._context(context)
        old = core._native_options(options)
        n,k,r,d,c = (refs[x] for x in ("nodes","cells","rank","dimension","classes"))
        core._matrix(z,(n,d),device=z.device); core._matrix(q,(n,c),device=z.device)
        _require(not z.requires_grad and not q.requires_grad and bool((q>=0).all())
            and bool((q.sum(1)>0).all()) and bool((q.sum(0)>0).all())
            and z.device.type in ("cpu","cuda"), "Raw frozen Q/z domain changed")
        _require(torch.is_tensor(assignment) and tuple(assignment.shape)==(n,)
            and assignment.dtype==torch.int64 and assignment.device==z.device
            and not assignment.requires_grad and int(assignment.min())>=0 and int(assignment.max())==k-1,
            "Native hard-cell order changed")
        data=array_digest(z.detach().cpu().numpy(),q.detach().cpu().numpy(),assignment.cpu().numpy())
        _require(data==refs["data_digest"] and old==refs["original_options"]
            and ("data_digest" not in options or options["data_digest"]==data)
            and refs["device"]==str(z.device) and options["assignment_rank"]==r
            and options["factor_seed"]==refs["factor_seed"] and options["chunk_size"]==refs["chunk_size"]
            and core._runtime(z.device)==refs["runtime"] and torch.get_default_dtype()==torch.float32
            and not torch.is_autocast_enabled(z.device.type), "Source/options/runtime changed")
        _require(all(_tensor_identity(value)==assets[name] for name,value in (("z",z),("Q",q),("assignment",assignment)))
            and core._factor_digests(initial,n,k,r,device=z.device)==context["native_parameter_digests"]
            and bool(initial[0].eq(0).all()), "Exact native source/U0/V0 identity changed")
        _require(self.rows.shape==(n,basis) and self.rows.descriptor()==self.execution["rows"], "Frozen original Phi descriptor changed")
        paths,pins=refs.get("asset_paths"),refs.get("files_sha256")
        _require(isinstance(paths,dict) and set(paths)=={"H","map","Phi","Phi_metadata"}
            and isinstance(pins,dict) and all(type(p) is str and p in pins for p in paths.values())
            and Path(paths["Phi_metadata"]).is_file(), "Mandatory original assets/sidecar unpinned")
        core._files(pins)
        _require(json.loads(Path(paths["Phi_metadata"]).read_text())==assets["Phi_identity"], "Original Phi sidecar changed")
        source=refs.get("current_source")
        _require(isinstance(source,dict) and type(source.get("git_head")) is str
            and type(source.get("source_digest")) is str, "Current source unbound")
        core._files(source.get("files"))
        _require(refs.get("optimizer_source_sha256")==hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(), "Current optimizer source differs")
        core._files({x["path"]:x["sha256"] for x in
            (self.execution["provider_source"],self.execution["operator_source"],self.execution["arithmetic_acceptance"])})
        _require(Path(self.execution["provider_source"]["path"]).resolve()==Path(__file__).resolve(), "Unexpected provider implementation")
        _require(Path(self.execution["operator_source"]["path"]).resolve()==Path(rowtile.__file__).resolve(), "Unexpected row-tile arithmetic implementation")
        acceptance=json.loads(Path(self.execution["arithmetic_acceptance"]["path"]).read_text())
        _require(isinstance(acceptance,dict) and (acceptance.get("passed") is True
            or acceptance.get("root_acceptance") is True), "Pinned prerequisite arithmetic acceptance is not passed")
        self.rows.assert_immutable(full=True,evidence=self.source_validation,stop=self.stop)
        self.admitted=True
        self.check()
        return expected_config(old,context)

    def _event(self, kind, step):
        _require(self.admitted and type(step) is int and step>=0, "Execution has no admitted native source/step")
        self.check()
        event=dict(index=len(self.events),step=step,kind=kind,evidence={})
        self.events.append(event)
        return event["evidence"]

    def critic_moments(self, U, V, assignment, q, step):
        ev=self._event("critic_moments",step)
        return TiledKernelMeanMoments.apply(U,V,assignment,self.rows,q,.05,
            self.context["source_refs"]["chunk_size"],ev,self.stop)

    def outer(self, q, theta, step):
        ev=self._event("source_outer",step)
        return bounded_outer_gradient(self.rows,q,theta,
            self.context["source_refs"]["original_options"]["outer_chunk_size"],ev,self.stop)

    def checkpoint_state(self, step, work):
        events=_json([{**event,"evidence":{k:v for k,v in event["evidence"].items() if k!="partial_arrays"}} for event in self.events])
        state=dict(schema=SCHEMA,execution=self.execution,step=step,events=events,summary=_summary(events))
        state["state_digest"]=core._seal(state)
        return validate_execution_state(state,self.context,step,work)

    def restore(self, saved, step, work):
        _require(not self.events and self.admitted, "Only a fresh admitted provider can consume a validated prefix")
        validate_execution_state(saved,self.context,step,work)
        self.events=_json(saved["events"])

    def failure_evidence(self):
        """Owning raw partials for exclusive failure artifact; never accepted state."""
        return core._cpu(dict(execution=self.execution,events=self.events,
            source_validation=self.source_validation,elapsed_seconds=time.monotonic()-self.started,
            failed_or_partial=True))

    def finish(self):
        self.check()
        self.rows.assert_immutable(full=True,evidence=self.source_validation,stop=self.stop)
        refs=self.context["source_refs"]
        core._files(refs["files_sha256"]);core._files(refs["current_source"]["files"])
        _require(core._runtime(torch.device(refs["device"]))==refs["runtime"], "Final arithmetic runtime changed")
        self.check()

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .session import WorkbenchSession
from .types import InterventionMode, InterventionSpec, PoleType


# =============================================================================
# 1. Pydantic Request / Response Models
# =============================================================================
class AuditRequest(BaseModel):
    layer_names: List[str] = Field(..., description="Target neuron layer names to audit")
    prompts: List[str] = Field(..., min_length=1, description="Corpus of text prompts to stream")
    top_k: int = Field(default=8, ge=1, le=50)
    tracked_neurons: Optional[List[int]] = Field(default=None)


class InterventionRequest(BaseModel):
    layer_name: str = Field(..., description="Target layer name (e.g. mlp_block_0.down_proj)")
    neuron_indices: List[int] = Field(..., min_length=1, description="Neuron indices to manipulate")
    mode: InterventionMode = Field(default=InterventionMode.ZERO_ABLATE)
    pole: PoleType = Field(default=PoleType.BOTH)
    value: float = Field(default=0.0)
    per_neuron_values: Optional[List[float]] = Field(default=None)
    token_positions: Optional[List[int]] = Field(default=None)


class CircuitTraceRequest(BaseModel):
    src_layer: str
    dst_layer: str
    src_neuron: int = Field(..., ge=0)
    src_pole: PoleType = Field(default=PoleType.POSITIVE)
    prompt: str = Field(..., description="Input prompt used to activate the source pole")
    compute_jacobian: bool = Field(default=True)
    top_k: int = Field(default=8, ge=1, le=64)


class EvaluateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to compare Clean vs. Surgically Intervened outputs")
    top_k_logits: int = Field(default=6, ge=1, le=20)


class ExplainRequest(BaseModel):
    layer_name: str
    neuron_idx: int = Field(..., ge=0)
    dst_layer: Optional[str] = Field(default=None)
    circuit_prompt: Optional[str] = Field(default="the system negotiated AES256 TLS cipher")


class ExportPresetRequest(BaseModel):
    preset_name: str = Field(default="crypto_sql_steering_v1")
    description: str = Field(default="Exported from Neural Injector Workbench")
    save_json_path: Optional[str] = Field(default=None)
    save_pt_vectors_path: Optional[str] = Field(default=None)


class ImportPresetRequest(BaseModel):
    preset_payload: Dict[str, Any]
    replace_existing: bool = Field(default=True)


# =============================================================================
# 2. FastAPI Factory & Thread-Safe, Path-Hardened Endpoints (Sections 4.1 & 4.2)
# =============================================================================
def create_workbench_app(
    session: Optional[WorkbenchSession] = None,
    allowed_workspace_dir: Optional[Union[str, Path]] = None,
) -> FastAPI:
    wb_session = session or WorkbenchSession.create_demo_session()
    if allowed_workspace_dir is not None:
        wb_session.injector.allowed_workspace_dir = Path(allowed_workspace_dir).resolve()
    elif wb_session.injector.allowed_workspace_dir is None:
        wb_session.injector.allowed_workspace_dir = Path.cwd().resolve()

    app = FastAPI(
        title="Neural Injector Workbench API",
        description="Bipolar Neuron Auditor, Surgical Activation Injector, Circuit Tracer & Agentic Explainer",
        version="0.6.0",
    )
    session_lock = asyncio.Lock()
    app_state = getattr(app, "state", None)
    if app_state is not None:
        setattr(app_state, "session", wb_session)
        setattr(app_state, "lock", session_lock)

    @app.get("/api/layers")
    async def list_layers(include_all_linear: bool = False) -> Dict[str, Any]:
        layers = wb_session.injector.list_targetable_layers(include_all_linear=include_all_linear)
        return {
            "model_name": wb_session.model_name,
            "vocab_size": len(wb_session.tok_to_id),
            "layers": [
                {
                    "name": desc.name,
                    "module_type": desc.module_type,
                    "num_neurons": desc.num_neurons,
                    "hook_target": desc.hook_target,
                    "audited": wb_session.injector.is_layer_audited(desc.name),
                }
                for desc in layers.values()
            ],
        }

    @app.post("/api/audit")
    async def audit_layers(req: AuditRequest) -> Dict[str, Any]:
        async with session_lock:
            try:
                return wb_session.run_audit(
                    layer_names=req.layer_names,
                    prompts=req.prompts,
                    top_k=req.top_k,
                    tracked_neurons=req.tracked_neurons,
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/layers/{layer_name}/scan")
    async def scan_layer_cleanliness(
        layer_name: str,
        sort_by: str = "pole_selectivity_score",
        compute_kl_top_n: int = 3,
        limit: int = 64,
    ) -> Dict[str, Any]:
        async with session_lock:
            try:
                kl_inputs, _ = wb_session.encode_prompt("the system negotiated AES256 TLS cipher")
                report = wb_session.injector.scan_layer_cleanliness(
                    layer_name=layer_name,
                    sort_by=sort_by,
                    compute_kl_for_top_n=compute_kl_top_n,
                    kl_eval_inputs=kl_inputs if compute_kl_top_n > 0 else None,
                    limit=limit,
                )
                return asdict(report)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/layers/{layer_name}/neurons/{neuron_idx}")
    async def get_neuron_profile(layer_name: str, neuron_idx: int) -> Dict[str, Any]:
        try:
            profile = wb_session.injector.get_neuron_profile(
                layer_name=layer_name,
                neuron_idx=neuron_idx,
                unembed_weight=wb_session.unembed_weight,
                idx_to_token=wb_session.id_to_tok,
            )
            data = asdict(profile)
            data["is_clean_candidate"] = profile.is_clean_candidate
            return data
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(
                status_code=400, detail=f"Neuron index {neuron_idx} out of range."
            ) from exc

    @app.post("/api/interventions")
    async def add_intervention(req: InterventionRequest) -> Dict[str, Any]:
        async with session_lock:
            spec = InterventionSpec(
                layer_name=req.layer_name,
                neuron_indices=req.neuron_indices,
                mode=req.mode,
                pole=req.pole,
                value=req.value,
                per_neuron_values=req.per_neuron_values,
                token_positions=req.token_positions,
            )
            wb_session.injector.add_intervention(spec)
            return {
                "status": "registered",
                "active_count": wb_session.injector.active_intervention_count,
                "intervention": asdict(spec),
            }

    @app.delete("/api/interventions")
    async def clear_interventions() -> Dict[str, Any]:
        async with session_lock:
            wb_session.injector.clear_interventions()
            return {"status": "cleared", "active_count": 0}

    @app.post("/api/evaluate")
    async def evaluate_prompt(req: EvaluateRequest) -> Dict[str, Any]:
        async with session_lock:
            return wb_session.evaluate_prompt_comparison(
                prompt=req.prompt, top_k_logits=req.top_k_logits
            )

    @app.post("/api/circuits/trace")
    async def trace_circuit(req: CircuitTraceRequest) -> Dict[str, Any]:
        async with session_lock:
            try:
                input_ids, _ = wb_session.encode_prompt(req.prompt)
                graph = wb_session.injector.trace_circuit(
                    src_layer=req.src_layer,
                    dst_layer=req.dst_layer,
                    src_neuron=req.src_neuron,
                    inputs=input_ids,
                    src_pole=req.src_pole,
                    compute_jacobian=req.compute_jacobian,
                    top_k=req.top_k,
                )
                return graph.to_agent_payload()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/explain")
    async def explain_neuron(req: ExplainRequest) -> Dict[str, Any]:
        async with session_lock:
            try:
                circuit_inputs = None
                if req.dst_layer and req.circuit_prompt:
                    circuit_inputs, _ = wb_session.encode_prompt(req.circuit_prompt)

                hypothesis = wb_session.injector.explain_neuron(
                    layer_name=req.layer_name,
                    neuron_idx=req.neuron_idx,
                    tok_to_id=wb_session.tok_to_id,
                    idx_to_token=wb_session.id_to_tok,
                    unembed_weight=wb_session.unembed_weight,
                    dst_layer=req.dst_layer,
                    circuit_inputs=circuit_inputs,
                )
                return asdict(hypothesis)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/export/preset")
    async def export_preset(req: ExportPresetRequest) -> Dict[str, Any]:
        async with session_lock:
            try:
                bundle = wb_session.injector.export_production_preset(
                    preset_name=req.preset_name,
                    model_name=wb_session.model_name,
                    description=req.description,
                    save_json_path=req.save_json_path,
                    save_pt_vectors_path=req.save_pt_vectors_path,
                )
                return asdict(bundle)
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/api/import/preset")
    async def import_preset(req: ImportPresetRequest) -> Dict[str, Any]:
        async with session_lock:
            bundle = wb_session.injector.load_production_preset(
                preset_data_or_path=req.preset_payload,
                replace_existing=req.replace_existing,
            )
            return {
                "status": "imported",
                "preset_name": bundle.preset_name,
                "active_interventions": len(bundle.interventions),
            }

    @app.post("/api/export/fold-weights")
    async def fold_weights() -> Dict[str, Any]:
        async with session_lock:
            return wb_session.injector.fold_interventions_into_weights()

    @app.get("/", response_class=HTMLResponse)
    async def workbench_ui() -> HTMLResponse:
        return HTMLResponse(content=WORKBENCH_HTML)

    return app


# =============================================================================
# 3. Embedded Interactive Browser Workbench UI
# =============================================================================
WORKBENCH_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <title>Neural Injector Workbench</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans p-6">
  <div class="max-w-7xl mx-auto space-y-6">
    <header class="flex items-center justify-between border-b border-slate-800 pb-4">
      <div>
        <h1 class="text-2xl font-bold text-emerald-400">Neural Injector Workbench v0.6</h1>
        <p class="text-sm text-slate-400">Bipolar Auditor • Layer-Wide Cleanliness Scanner • Surgical Injector • Circuit Tracer • Production Steering Exporter</p>
      </div>
      <span id="model-badge" class="px-3 py-1 rounded bg-slate-800 text-xs font-mono text-emerald-300">Loading...</span>
    </header>

    <!-- Row 1: Layer-Wide Cleanliness Scanner & Heatmap -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <div class="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 class="text-lg font-semibold text-indigo-400">1. Batch Layer-Wide Cleanliness Scanner</h2>
          <p class="text-xs text-slate-400">Scans all neurons &amp; poles in parallel using GPU Welford 4th-moment kurtosis &amp; outgoing residual norms. Click any neuron tile to inspect it.</p>
        </div>
        <div class="flex items-center gap-2">
          <select id="scan-sort" class="bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs">
            <option value="pole_selectivity_score">Sort: Pole Selectivity</option>
            <option value="estimated_causal_impact">Sort: Causal Impact</option>
            <option value="bipolar_balance">Sort: Bipolar Balance</option>
            <option value="neuron_idx">Sort: Neuron Index</option>
          </select>
          <button onclick="scanLayerCleanliness()" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded">Scan Layer Cleanliness</button>
        </div>
      </div>
      <div id="scan-metrics" class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs"></div>
      <div id="scan-heatmap" class="flex flex-wrap gap-1.5 pt-2"></div>
    </div>

    <!-- Row 2: Inspector, Surgical Injector, Circuit Tracer -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 class="text-lg font-semibold text-sky-400">2. Bipolar Neuron Inspector</h2>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="text-xs text-slate-400">Layer</label>
            <select id="layer-select" onchange="scanLayerCleanliness()" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm"></select>
          </div>
          <div>
            <label class="text-xs text-slate-400">Neuron Index</label>
            <input id="neuron-idx" type="number" value="7" min="0" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm" />
          </div>
        </div>
        <div class="flex gap-2">
          <button onclick="inspectNeuron()" class="flex-1 bg-sky-600 hover:bg-sky-500 text-white text-sm font-medium py-2 rounded">Inspect Poles</button>
          <button onclick="explainNeuron()" class="flex-1 bg-purple-600 hover:bg-purple-500 text-white text-sm font-medium py-2 rounded">Agent Explainer</button>
        </div>
        <pre id="profile-output" class="bg-slate-950 p-3 rounded text-xs font-mono overflow-auto max-h-80 text-slate-300"></pre>
      </div>

      <!-- Panel 3: Surgical Intervention & Live Evaluation -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 class="text-lg font-semibold text-amber-400">3. Surgical Neuron Injector</h2>
        <div class="grid grid-cols-3 gap-2">
          <div>
            <label class="text-xs text-slate-400">Mode</label>
            <select id="int-mode" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs">
              <option value="zero_ablate">zero_ablate (=0)</option>
              <option value="hard_clamp">hard_clamp (=c)</option>
              <option value="additive_steer">additive_steer (+d)</option>
              <option value="scale">scale (*alpha)</option>
              <option value="weak_prune">weak_prune (|a|&lt;t)</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-slate-400">Target Pole</label>
            <select id="int-pole" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs">
              <option value="both">both</option>
              <option value="positive">positive (+)</option>
              <option value="negative">negative (-)</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-slate-400">Value</label>
            <input id="int-val" type="number" step="0.5" value="-15.0" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs" />
          </div>
        </div>
        <div class="flex gap-2">
          <button onclick="addIntervention()" class="flex-1 bg-amber-600 hover:bg-amber-500 text-white text-sm font-medium py-2 rounded">Inject Intervention</button>
          <button onclick="clearInterventions()" class="bg-rose-700 hover:bg-rose-600 text-white text-sm font-medium px-3 py-2 rounded">Clear</button>
        </div>
        <div>
          <label class="text-xs text-slate-400">Test Prompt</label>
          <input id="eval-prompt" type="text" value="the system negotiated AES256" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm mt-1" />
        </div>
        <button onclick="evaluatePrompt()" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium py-2 rounded">Compare Clean vs. Intervened</button>
        <pre id="eval-output" class="bg-slate-950 p-3 rounded text-xs font-mono overflow-auto max-h-64 text-slate-300"></pre>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
        <h2 class="text-lg font-semibold text-emerald-400">4. Circuit Tracer &amp; Steering Exporter</h2>
        <div class="grid grid-cols-2 gap-2">
          <div>
            <label class="text-xs text-slate-400">Destination Layer</label>
            <select id="dst-layer-select" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm"></select>
          </div>
          <div>
            <label class="text-xs text-slate-400">Source Pole</label>
            <select id="trace-pole" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm">
              <option value="positive">positive (+)</option>
              <option value="negative">negative (-)</option>
            </select>
          </div>
        </div>
        <div class="flex gap-2">
          <button onclick="traceCircuit()" class="flex-1 bg-teal-600 hover:bg-teal-500 text-white text-xs font-medium py-2 rounded">Trace Circuit</button>
          <button onclick="exportPreset()" class="flex-1 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium py-2 rounded">Export Steering Preset</button>
          <button onclick="foldWeights()" class="bg-fuchsia-700 hover:bg-fuchsia-600 text-white text-xs font-medium px-2.5 py-2 rounded" title="Fold active interventions into down_proj weights/bias">Fold Weights</button>
        </div>
        <pre id="circuit-output" class="bg-slate-950 p-3 rounded text-xs font-mono overflow-auto max-h-80 text-slate-300"></pre>
      </div>
    </div>
  </div>

  <script>
    async function initWorkbench() {
      const res = await fetch('/api/layers');
      const data = await res.json();
      document.getElementById('model-badge').textContent = `${data.model_name} (${data.vocab_size} vocab)`;
      const srcSel = document.getElementById('layer-select');
      const dstSel = document.getElementById('dst-layer-select');
      srcSel.innerHTML = ''; dstSel.innerHTML = '';
      data.layers.forEach((l) => {
        srcSel.add(new Option(`${l.name} (${l.num_neurons}n)`, l.name));
        dstSel.add(new Option(`${l.name} (${l.num_neurons}n)`, l.name));
      });
      if (data.layers.length > 1) dstSel.selectedIndex = 1;
      await scanLayerCleanliness();
      await inspectNeuron();
    }

    async function scanLayerCleanliness() {
      const layer = document.getElementById('layer-select').value;
      const sortBy = document.getElementById('scan-sort').value;
      const res = await fetch(`/api/layers/${encodeURIComponent(layer)}/scan?sort_by=${sortBy}&compute_kl_top_n=3&limit=64`);
      const data = await res.json();

      document.getElementById('scan-metrics').innerHTML = `
        <div class="bg-slate-950 p-3 rounded border border-slate-800"><span class="text-slate-400 block">Clean Monosemantic</span><span class="text-lg font-bold text-emerald-400">${data.clean_monosemantic_pct}%</span></div>
        <div class="bg-slate-950 p-3 rounded border border-slate-800"><span class="text-slate-400 block">Clean Poles (+ / -)</span><span class="text-lg font-bold text-sky-400">${data.clean_poles_pct}%</span></div>
        <div class="bg-slate-950 p-3 rounded border border-slate-800"><span class="text-slate-400 block">Combined Clean + Structured</span><span class="text-lg font-bold text-indigo-400">${data.combined_clean_and_structured_pct}%</span></div>
        <div class="bg-slate-950 p-3 rounded border border-slate-800"><span class="text-slate-400 block">Mean Selectivity Score</span><span class="text-lg font-bold text-amber-400">${data.mean_selectivity_score}</span></div>
      `;

      const heatmap = document.getElementById('scan-heatmap');
      heatmap.innerHTML = '';
      data.neurons.forEach(n => {
        let color = 'bg-slate-800 border-slate-700 text-slate-300';
        if (n.taxonomy_class === 'clean_monosemantic') color = 'bg-emerald-950 border-emerald-600 text-emerald-300';
        else if (n.taxonomy_class === 'structured_recursive_mixture') color = 'bg-sky-950 border-sky-600 text-sky-300';
        else if (n.taxonomy_class === 'true_polysemantic') color = 'bg-amber-950 border-amber-600 text-amber-300';

        const btn = document.createElement('button');
        btn.className = `px-2 py-1 rounded border text-xs font-mono hover:scale-105 transition ${color}`;
        btn.title = `N#${n.neuron_idx} | ${n.taxonomy_class} | Selectivity=${n.pole_selectivity_score} | Impact=${n.estimated_causal_impact}`;
        btn.textContent = `N${n.neuron_idx} (${n.pole_selectivity_score.toFixed(1)})`;
        btn.onclick = () => {
          document.getElementById('neuron-idx').value = n.neuron_idx;
          inspectNeuron();
        };
        heatmap.appendChild(btn);
      });
    }

    async function inspectNeuron() {
      const layer = document.getElementById('layer-select').value;
      const idx = document.getElementById('neuron-idx').value;
      const res = await fetch(`/api/layers/${encodeURIComponent(layer)}/neurons/${idx}`);
      document.getElementById('profile-output').textContent = JSON.stringify(await res.json(), null, 2);
    }

    async function explainNeuron() {
      const layer = document.getElementById('layer-select').value;
      const dstLayer = document.getElementById('dst-layer-select').value;
      const idx = parseInt(document.getElementById('neuron-idx').value);
      const prompt = document.getElementById('eval-prompt').value;
      const res = await fetch('/api/explain', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({layer_name: layer, neuron_idx: idx, dst_layer: dstLayer, circuit_prompt: prompt})
      });
      document.getElementById('profile-output').textContent = JSON.stringify(await res.json(), null, 2);
    }

    async function addIntervention() {
      const layer = document.getElementById('layer-select').value;
      const idx = parseInt(document.getElementById('neuron-idx').value);
      const mode = document.getElementById('int-mode').value;
      const pole = document.getElementById('int-pole').value;
      const val = parseFloat(document.getElementById('int-val').value);
      await fetch('/api/interventions', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({layer_name: layer, neuron_indices: [idx], mode, pole, value: val})
      });
      evaluatePrompt();
    }

    async function clearInterventions() {
      await fetch('/api/interventions', {method: 'DELETE'});
      evaluatePrompt();
    }

    async function evaluatePrompt() {
      const prompt = document.getElementById('eval-prompt').value;
      const res = await fetch('/api/evaluate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({prompt, top_k_logits: 5})
      });
      document.getElementById('eval-output').textContent = JSON.stringify(await res.json(), null, 2);
    }

    async function traceCircuit() {
      const srcLayer = document.getElementById('layer-select').value;
      const dstLayer = document.getElementById('dst-layer-select').value;
      const idx = parseInt(document.getElementById('neuron-idx').value);
      const pole = document.getElementById('trace-pole').value;
      const prompt = document.getElementById('eval-prompt').value;
      const res = await fetch('/api/circuits/trace', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({src_layer: srcLayer, dst_layer: dstLayer, src_neuron: idx, src_pole: pole, prompt})
      });
      document.getElementById('circuit-output').textContent = JSON.stringify(await res.json(), null, 2);
    }

    async function exportPreset() {
      const res = await fetch('/api/export/preset', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({preset_name: 'workbench_steering_preset'})
      });
      document.getElementById('circuit-output').textContent = JSON.stringify(await res.json(), null, 2);
    }

    async function foldWeights() {
      const res = await fetch('/api/export/fold-weights', {method: 'POST'});
      document.getElementById('circuit-output').textContent = JSON.stringify(await res.json(), null, 2);
      evaluatePrompt();
    }

    window.onload = initWorkbench;
  </script>
</body>
</html>
"""

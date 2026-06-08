#!/usr/bin/env python3
"""Generate interactive stochastic-inference pool viewer HTML."""

import ast
import json
import math
import re
from pathlib import Path

EVENTS_FILE = "reaction_stochastic_inference_transformer/transformer_decoder/events.txt"
OUT_HTML    = "transformer_decoder_inference_pool_viewer.html"

# ── molecule geometry (normalized coords) ─────────────────────────────────────
MOLECULE_DATA = {
    "H":    {"atoms":["H"],                    "adj":[[0]],                                     "pos":[[0,0]]},
    "O":    {"atoms":["O"],                    "adj":[[0]],                                     "pos":[[0,0]]},
    "H2":   {"atoms":["H","H"],                "adj":[[0,1],[1,0]],                             "pos":[[-0.6,0],[0.6,0]]},
    "O2":   {"atoms":["O","O"],                "adj":[[0,2],[2,0]],                             "pos":[[-0.6,0],[0.6,0]]},
    "OH":   {"atoms":["O","H"],                "adj":[[0,1],[1,0]],                             "pos":[[-0.5,0],[0.5,0]]},
    "H2O":  {"atoms":["O","H","H"],            "adj":[[0,1,1],[1,0,0],[1,0,0]],                "pos":[[0,0.35],[-0.7,-0.5],[0.7,-0.5]]},
    "HO2":  {"atoms":["H","O","O"],            "adj":[[0,1,0],[1,0,1],[0,1,0]],                "pos":[[-1.1,0],[0,0],[0.9,0.7]]},
    "H2O2": {"atoms":["H","O","O","H"],        "adj":[[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]],"pos":[[-1.5,0.6],[-0.5,0],[0.5,0],[1.5,0.6]]},
    "UNK_0001": {"atoms":["O","O"],            "adj":[[0,1],[1,0]],                             "pos":[[-0.6,0],[0.6,0]], "unknown":True},
    "UNK_0002": {"atoms":["O","O","O"],        "adj":[[0,1,1],[1,0,1],[1,1,0]],                "pos":[[0,0.8],[-0.7,-0.4],[0.7,-0.4]], "unknown":True},
}

SPECIES_LABEL = {
    "H":"H","O":"O","H2":"H₂","O2":"O₂","OH":"OH",
    "H2O":"H₂O","HO2":"HO₂","H2O2":"H₂O₂",
    "UNK_0001":"UNK_0001","UNK_0002":"UNK_0002",
}

DISPLAY_ORDER = ["H","O","H2","O2","OH","H2O","HO2","H2O2","UNK_0001","UNK_0002"]

# ── parse events.txt ──────────────────────────────────────────────────────────
PAT = re.compile(
    r"step (\d+): (ACCEPT|REJECT) reactants=(\[.*?\]) "
    r"third_body=(\d) branch=\S+ p=([\d.]+) channels=(\d+) "
    r"-> products=(.*?) pool=(\{.*\})"
)

def parse_events(path: str):
    steps = [{
        "step":0,"event":"initial","reactants":[],"products":[],
        "third_body":False,"channels":0,"prob":0.0,
        "pool":{"H2":15,"O2":15},
        "desc":"Initial pool: H₂×15, O₂×15",
    }]
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = PAT.match(line.strip())
            if not m:
                continue
            sn      = int(m.group(1))
            ev      = m.group(2)
            rcts    = ast.literal_eval(m.group(3))
            tb      = bool(int(m.group(4)))
            prob    = float(m.group(5))
            ch      = int(m.group(6))
            prod_s  = m.group(7).strip()
            pool    = ast.literal_eval(m.group(8))

            if prod_s == "UNKNOWN":
                prods = []
                ev_label = f"REJECT (unknown products)"
            else:
                prods = ast.literal_eval(prod_s)
                ev_label = ev

            tb_tag = " [+M]" if tb else ""
            r_str  = " + ".join(rcts) if rcts else "—"
            p_str  = " + ".join(prods) if prods else ("UNKNOWN" if ev == "REJECT" else "—")
            desc   = f"{ev_label}: {r_str} → {p_str}{tb_tag}  (p={prob:.3f}, ch={ch})"

            steps.append({
                "step":sn,"event":ev,
                "reactants":list(rcts),"products":list(prods) if prods else [],
                "third_body":tb,"channels":ch,"prob":prob,
                "pool":dict(pool),"desc":desc,
            })
    return steps

# ── compute species history for chart ─────────────────────────────────────────
def build_species_history(steps):
    all_sp = []
    seen = set()
    for s in steps:
        for sp in s["pool"]:
            if sp not in seen:
                seen.add(sp)
                all_sp.append(sp)
    def order_key(sp):
        if sp in DISPLAY_ORDER:
            return DISPLAY_ORDER.index(sp)
        return len(DISPLAY_ORDER)
    all_sp.sort(key=order_key)
    history = {sp: [int(s["pool"].get(sp,0)) for s in steps] for sp in all_sp}
    return all_sp, history

# ── HTML generation ────────────────────────────────────────────────────────────
SP_COLORS = {"#59a14f","#4e79a7","#e15759","#f28e2b","#76b7b2","#59a14f",
             "#edc948","#b07aa1","#ff9da7","#9c755f"}
PLOTLY_PALETTE = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
                  "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"]

def generate_html(steps, out_path):
    all_sp, sp_hist = build_species_history(steps)
    n_steps = len(steps) - 1  # 0..600

    steps_json = json.dumps(steps, separators=(',',':'))
    mol_json   = json.dumps(MOLECULE_DATA, separators=(',',':'))
    all_sp_json= json.dumps(all_sp)
    sp_hist_json = json.dumps(sp_hist, separators=(',',':'))
    sp_label_json = json.dumps(SPECIES_LABEL)
    display_order_json = json.dumps(DISPLAY_ORDER)
    palette_json = json.dumps(PLOTLY_PALETTE)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stochastic Inference — Pool Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0f1117;color:#e0e0e0;padding:16px}}
h1{{text-align:center;color:#7eb8f7;font-size:1.3em;margin-bottom:4px}}
.subtitle{{text-align:center;color:#888;font-size:.82em;margin-bottom:14px}}

/* controls */
.controls{{display:flex;align-items:center;gap:12px;justify-content:center;
           flex-wrap:wrap;margin-bottom:12px}}
select,button{{background:#2a2d3a;color:#e0e0e0;border:1px solid #3a3d4a;
               padding:5px 12px;border-radius:6px;cursor:pointer;font-size:.85em}}
button:hover{{background:#3a3d4a}}
.step-badge{{font-size:1.05em;font-weight:bold;color:#7eb8f7;
             background:#1a2a3a;padding:3px 14px;border-radius:20px;
             border:1px solid #3a5a7a;min-width:90px;text-align:center}}
input[type=range]{{width:340px;accent-color:#7eb8f7}}

/* event bar */
.event-bar{{background:#1a1d27;border:1px solid #2a2d3a;border-radius:8px;
            padding:8px 16px;text-align:center;font-size:.88em;color:#ccc;
            max-width:860px;margin:0 auto 12px;min-height:34px}}
.ev-accept{{color:#59a14f;font-weight:bold}}
.ev-reject{{color:#e15759;font-weight:bold}}
.ev-initial{{color:#7eb8f7;font-weight:bold}}

/* pool area */
.pool-label{{text-align:center;font-size:.8em;color:#666;margin-bottom:6px}}
.pool-row{{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;
           max-width:1100px;margin:0 auto 14px;min-height:160px;
           align-items:flex-start}}
.sp-card{{background:#1a1d27;border-radius:10px;border:1px solid #2a2d3a;
          padding:8px;text-align:center;width:130px;position:relative}}
.sp-card.unknown{{border-color:#e15759;background:#1f1515}}
.sp-card h5{{font-size:.82em;color:#a0c4ff;margin-bottom:4px}}
.sp-card h5.unk-lbl{{color:#e15759}}
.sp-card svg{{width:100%;height:90px;display:block}}
.sp-count{{position:absolute;top:6px;right:8px;background:#3a3d4a;
           color:#e0e0e0;font-size:.75em;font-weight:bold;
           padding:1px 7px;border-radius:10px}}
.sp-card.unknown .sp-count{{background:#5a2020}}

/* bottom grid */
.bottom{{display:grid;grid-template-columns:1fr 1fr;gap:14px;
         max-width:1100px;margin:0 auto}}
.card{{background:#1a1d27;border-radius:10px;border:1px solid #2a2d3a;padding:12px}}
.card h4{{margin:0 0 8px;color:#a0c4ff;font-size:.85em}}
#speciesChart{{height:260px}}
#poolChart{{height:260px}}
</style>
</head>
<body>
<h1>Stochastic Inference — Pool Evolution</h1>
<p class="subtitle">GNN+Transformer model · {n_steps} steps · T=300 K · P=0.1 atm · start: H₂×15, O₂×15</p>

<div class="controls">
  <button id="playBtn" onclick="togglePlay()">▶ Play</button>
  <input type="range" id="stepSlider" min="0" max="{n_steps}" value="0" oninput="onSlider()">
  <div class="step-badge" id="stepBadge">Step 0</div>
  <button onclick="jumpTo(0)">⏮ Reset</button>
  <button onclick="jumpTo({n_steps})">⏭ Final</button>
</div>

<div class="event-bar" id="eventBar">Initial pool</div>
<p class="pool-label">Current pool — one card per species, count shown top-right</p>
<div class="pool-row" id="poolRow"></div>

<div class="bottom">
  <div class="card">
    <h4>Species Counts vs Step</h4>
    <div id="speciesChart"></div>
  </div>
  <div class="card">
    <h4>Pool Composition at Current Step</h4>
    <div id="poolChart"></div>
  </div>
</div>

<script>
const STEPS = {steps_json};
const MOL   = {mol_json};
const ALL_SP = {all_sp_json};
const SP_HIST = {sp_hist_json};
const SP_LABEL = {sp_label_json};
const DISPLAY_ORDER = {display_order_json};
const PALETTE = {palette_json};

const NS = 'http://www.w3.org/2000/svg';
const ATOM_COLOR = {{H:'#4e79a7', O:'#e15759'}};
const cfg = {{responsive:true,displayModeBar:false}};
const lb = {{
  paper_bgcolor:'#1a1d27',plot_bgcolor:'#1a1d27',
  font:{{color:'#ccc',size:11}},margin:{{t:6,b:40,l:50,r:10}},
  xaxis:{{gridcolor:'#2a2d3a',zerolinecolor:'#333'}},
  yaxis:{{gridcolor:'#2a2d3a',zerolinecolor:'#333'}},
  legend:{{bgcolor:'#1a1d27',bordercolor:'#333',font:{{size:10}}}},
}};

let curIdx = 0, playing = false, playTimer = null;

// ── SVG helpers ──────────────────────────────────────────────────────────────
function svgEl(tag, attrs) {{
  const e = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k,v]) => e.setAttribute(k,v));
  return e;
}}

function scalePos(pos, W, H, pad) {{
  if (pos.length === 1) return [[W/2, H/2]];
  const xs=pos.map(p=>p[0]), ys=pos.map(p=>p[1]);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  const rX=maxX-minX||1, rY=maxY-minY||1;
  const sc=Math.min((W-pad*2)/rX,(H-pad*2)/rY)*0.82;
  const offX=W/2-(minX+maxX)/2*sc, offY=H/2+(minY+maxY)/2*sc;
  return pos.map(p=>[offX+p[0]*sc, offY-p[1]*sc]);
}}

function drawBond(svg, x1,y1,x2,y2, order, color) {{
  if(order<=0) return;
  const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy)||1;
  const nx=-dy/len*3.5, ny=dx/len*3.5;
  function line(ax,ay,bx,by) {{
    svg.appendChild(svgEl('line',{{x1:ax,y1:ay,x2:bx,y2:by,
      stroke:color,'stroke-width':2,'stroke-linecap':'round'}}));
  }}
  if(order===1) line(x1,y1,x2,y2);
  else if(order===2) {{
    line(x1+nx,y1+ny,x2+nx,y2+ny);
    line(x1-nx,y1-ny,x2-nx,y2-ny);
  }} else {{
    line(x1+nx*1.5,y1+ny*1.5,x2+nx*1.5,y2+ny*1.5);
    line(x1,y1,x2,y2);
    line(x1-nx*1.5,y1-ny*1.5,x2-nx*1.5,y2-ny*1.5);
  }}
}}

function drawMolSvg(dom, molData) {{
  dom.innerHTML = '';
  const W = dom.clientWidth||120, H = dom.clientHeight||90;
  const n = molData.atoms.length;
  const spos = scalePos(molData.pos, W, H, 18);
  const r = Math.max(11, Math.min(17, 70/Math.max(n,1)));

  // bonds
  for(var i=0;i<n;i++) for(var j=i+1;j<n;j++) {{
    const b = molData.adj[i][j];
    if(b<=0) continue;
    const x1=spos[i][0],y1=spos[i][1],x2=spos[j][0],y2=spos[j][1];
    const dx=x2-x1,dy=y2-y1,d=Math.sqrt(dx*dx+dy*dy)||1;
    drawBond(dom, x1+dx/d*r,y1+dy/d*r, x2-dx/d*r,y2-dy/d*r, b, '#778');
  }}

  // atoms
  for(var k=0;k<n;k++) {{
    const cx=spos[k][0], cy=spos[k][1];
    const t = molData.atoms[k];
    const fc = ATOM_COLOR[t]||'#888';
    const circ = document.createElementNS(NS,'circle');
    circ.setAttribute('cx',cx); circ.setAttribute('cy',cy);
    circ.setAttribute('r',r); circ.setAttribute('fill',fc);
    circ.setAttribute('stroke','#ddd'); circ.setAttribute('stroke-width',1.2);
    dom.appendChild(circ);
    const txt = document.createElementNS(NS,'text');
    txt.setAttribute('x',cx); txt.setAttribute('y',cy);
    txt.setAttribute('text-anchor','middle');
    txt.setAttribute('dominant-baseline','middle');
    txt.setAttribute('fill','#fff');
    txt.setAttribute('font-size', r*0.85);
    txt.setAttribute('font-weight','bold');
    txt.textContent = t;
    dom.appendChild(txt);
  }}
}}

// ── pool rendering ────────────────────────────────────────────────────────────
function orderedPool(pool) {{
  const keys = Object.keys(pool).filter(k=>pool[k]>0);
  keys.sort(function(a,b) {{
    const ia = DISPLAY_ORDER.indexOf(a), ib = DISPLAY_ORDER.indexOf(b);
    const ra = ia<0?999:ia, rb = ib<0?999:ib;
    return ra-rb;
  }});
  return keys;
}}

function renderPool(pool) {{
  const row = document.getElementById('poolRow');
  row.innerHTML = '';
  const keys = orderedPool(pool);
  if(keys.length===0) {{
    row.innerHTML='<p style="color:#666;margin:auto">Pool is empty</p>';
    return;
  }}
  keys.forEach(function(sp) {{
    const cnt = pool[sp];
    const molData = MOL[sp];
    const isUnk = !!(molData && molData.unknown);
    const card = document.createElement('div');
    card.className = 'sp-card' + (isUnk?' unknown':'');

    const lbl = document.createElement('h5');
    lbl.className = isUnk ? 'unk-lbl' : '';
    lbl.textContent = (SP_LABEL[sp]||sp) + (isUnk?' ?':'');
    card.appendChild(lbl);

    const badge = document.createElement('div');
    badge.className='sp-count';
    badge.textContent='×'+cnt;
    card.appendChild(badge);

    const svg = document.createElementNS(NS,'svg');
    svg.setAttribute('width','100%');
    svg.setAttribute('height','90');
    card.appendChild(svg);

    if(molData) {{
      requestAnimationFrame(function(){{ drawMolSvg(svg, molData); }});  // svg is the SVG DOM element
    }} else {{
      const txt = document.createElementNS(NS,'text');
      txt.setAttribute('x','50%'); txt.setAttribute('y','50%');
      txt.setAttribute('text-anchor','middle');
      txt.setAttribute('dominant-baseline','middle');
      txt.setAttribute('fill','#888'); txt.setAttribute('font-size','11');
      txt.textContent='(no structure)';
      svg.appendChild(txt);
    }}
    row.appendChild(card);
  }});
}}

// ── event bar ─────────────────────────────────────────────────────────────────
function renderEvent(s) {{
  const bar = document.getElementById('eventBar');
  let cls = 'ev-initial';
  if(s.event==='ACCEPT') cls='ev-accept';
  else if(s.event==='REJECT') cls='ev-reject';
  bar.innerHTML = '<span class="'+cls+'">'+s.desc+'</span>';
}}

// ── charts ────────────────────────────────────────────────────────────────────
function buildSpeciesChart() {{
  const xs = Array.from({{length:STEPS.length}},(_,i)=>i);
  const traces = ALL_SP.map(function(sp,i) {{
    return {{
      x:xs, y:SP_HIST[sp], mode:'lines', name:SP_LABEL[sp]||sp,
      line:{{color:PALETTE[i%PALETTE.length],width:1.8}},
    }};
  }});
  const shapes = [{{
    type:'line',xref:'x',yref:'paper',
    x0:0,x1:0,y0:0,y1:1,
    line:{{color:'#f4a261',width:2,dash:'dot'}},
  }}];
  Plotly.newPlot('speciesChart', traces,
    {{...lb,
      xaxis:{{...lb.xaxis,title:'Step'}},
      yaxis:{{...lb.yaxis,title:'Count'}},
      shapes:shapes,
    }}, cfg);
}}

function updateSpeciesChartMarker(idx) {{
  Plotly.relayout('speciesChart',{{'shapes[0].x0':idx,'shapes[0].x1':idx}});
}}

function renderPoolBar(pool) {{
  const keys = orderedPool(pool);
  const vals = keys.map(k=>pool[k]);
  const colors = keys.map(function(k) {{
    const i = ALL_SP.indexOf(k);
    return i>=0?PALETTE[i%PALETTE.length]:'#888';
  }});
  Plotly.react('poolChart',[{{
    x:keys.map(k=>SP_LABEL[k]||k),
    y:vals,
    type:'bar',
    marker:{{color:colors}},
    text:vals.map(String),
    textposition:'auto',
    hovertemplate:'%{{x}}: %{{y}}<extra></extra>',
  }}],{{
    ...lb,
    xaxis:{{...lb.xaxis,title:''}},
    yaxis:{{...lb.yaxis,title:'Count'}},
  }}, cfg);
}}

// ── main render ───────────────────────────────────────────────────────────────
function render() {{
  const s = STEPS[curIdx];
  document.getElementById('stepBadge').textContent = 'Step '+s.step;
  document.getElementById('stepSlider').value = curIdx;
  renderEvent(s);
  renderPool(s.pool);
  updateSpeciesChartMarker(curIdx);
  renderPoolBar(s.pool);
}}

function onSlider() {{
  curIdx = parseInt(document.getElementById('stepSlider').value);
  render();
}}

function jumpTo(idx) {{
  curIdx = idx;
  render();
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('playBtn').textContent = playing?'⏸ Pause':'▶ Play';
  if(playing) advance(); else clearTimeout(playTimer);
}}

function advance() {{
  if(!playing) return;
  curIdx = Math.min(curIdx+1, STEPS.length-1);
  render();
  if(curIdx >= STEPS.length-1) {{ playing=false; document.getElementById('playBtn').textContent='▶ Play'; return; }}
  playTimer = setTimeout(advance, curIdx<10?800:120);
}}

// ── init ──────────────────────────────────────────────────────────────────────
buildSpeciesChart();
render();
</script>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    steps = parse_events(os.path.join(base, EVENTS_FILE))
    print(f"Parsed {len(steps)} entries (step 0..{len(steps)-1})")
    generate_html(steps, os.path.join(base, OUT_HTML))

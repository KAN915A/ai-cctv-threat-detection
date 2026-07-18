/* AI CCTV Threat Monitor — browser edition.
 * Runs three YOLOv8 ONNX models client-side (ONNX Runtime Web): a COCO
 * context model plus an ensemble of two weapons models, then applies the
 * same tracking + threat-classification rules as the Python backend.
 */

'use strict';

// ---------------------------------------------------------------- config --
const GITHUB_MODEL_BASE =
  'https://cdn.jsdelivr.net/gh/KAN915A/ai-cctv-threat-detection@main/web/models/';

const MODEL_SIZE = 640;
const GENERAL_CONF = 0.40;
const NMS_IOU = 0.45;

// Weapon fusion: candidates are cross-checked against the COCO model and
// scene geometry before they count (the raw weapons models false-positive
// on laptops, pens, screens, whole-room boxes).
const WEAPON_CANDIDATE_CONF = 0.30;   // raw model threshold
const WEAPON_CONF_NEAR_PERSON = 0.55; // weapon overlapping a person
const WEAPON_CONF_ENSEMBLE = 0.45;    // both weapons models agree
const WEAPON_CONF_AGREE = 0.40;       // custom + COCO 'knife' agree
const WEAPON_CONF_ALONE = 0.70;       // no person anywhere near
const WEAPON_MAX_AREA_FRAC = 0.30;    // bigger than this = misfire
const WEAPON_PERSON_IOU_VETO = 0.50;  // box ≈ person box = misfire
const DISTRACTOR_IOU = 0.40;
const DISTRACTOR_MIN_CONF = 0.40;
const ENSEMBLE_IOU = 0.55;            // same-object match across models
const DISTRACTORS = new Set(['cell phone','remote','laptop','tv','keyboard',
  'mouse','book','bottle','cup','umbrella','toothbrush','hair drier','clock',
  'teddy bear','banana']);
const DANGEROUS = new Set(['baseball bat', 'scissors']);

const LOITER_SECONDS = 15;
const LOITER_RADIUS = 90;            // px in model space
const VEHICLE_LURK_SECONDS = 8;
const ZONE_DWELL_SECONDS = 5;        // must stay in zone before alerting
// Temporal vote: weapon must appear in >= WEAPON_VOTES of the last
// WEAPON_WINDOW frames before HIGH fires (kills one-frame flickers)
const WEAPON_WINDOW = 8;
const WEAPON_VOTES = 5;
const WEAPON_CRITICAL_SECONDS = 5;
// Altercation heuristic: two people moving fast in close quarters
const FIGHT_SPEED_PX = 140;          // px/s each person must exceed
const FIGHT_WINDOW = 6;
const FIGHT_VOTES = 3;
const ALERT_COOLDOWN_SECONDS = 30;

const LEVELS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
const LEVEL_COLORS = { LOW: '#ffb40b', MEDIUM: '#ff7800', HIGH: '#e6383c', CRITICAL: '#c832c8' };
const KIND_COLORS = { person: '#5ac85a', vehicle: '#3cb4e6', package: '#c8a078',
                      weapon: '#e6383c', danger: '#ff8c00' };

const COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];
const WEAPON_NAMES = ['guns', 'knife'];

const VEHICLES = new Set(['car', 'truck', 'bus', 'motorcycle', 'bicycle']);
const PACKAGES = new Set(['backpack', 'handbag', 'suitcase']);

function kindFor(label, fromWeaponModel) {
  if (fromWeaponModel) return 'weapon';
  if (label === 'person') return 'person';
  if (VEHICLES.has(label)) return 'vehicle';
  if (PACKAGES.has(label)) return 'package';
  if (DANGEROUS.has(label)) return 'danger';
  return 'other';   // kept for weapon fusion (distractor veto), not displayed
}

// ------------------------------------------------------------------- dom --
const $ = id => document.getElementById(id);
const canvas = $('canvas'), ctx = canvas.getContext('2d');
const video = document.createElement('video');
video.playsInline = true; video.muted = true;

let sessions = null;      // { general, weapons: [s1, s2] }
let running = false;
let stream = null;
let audioCtx = null, lastBeep = 0;

// ------------------------------------------------------------ model load --
async function fetchModel(name, onProgress) {
  const urls = [`models/${name}`, GITHUB_MODEL_BASE + name];
  let lastErr;
  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status} for ${url}`);
      const total = +res.headers.get('content-length') || 0;
      const reader = res.body.getReader();
      const chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        if (total && onProgress) onProgress(got / total);
      }
      const buf = new Uint8Array(got);
      let off = 0;
      for (const c of chunks) { buf.set(c, off); off += c.length; }
      return buf;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

async function loadModels() {
  const ph = $('placeholder');
  const providers = (navigator.gpu ? ['webgpu', 'wasm'] : ['wasm']);
  // Multi-threaded WASM needs cross-origin isolation (COOP/COEP headers,
  // served by web/server.py locally and vercel.json in production)
  if (crossOriginIsolated) {
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
  }
  const opts = { executionProviders: providers };
  const progress = {};
  const report = () => {
    const parts = Object.entries(progress)
      .map(([n, p]) => `${n} ${(p * 100).toFixed(0)}%`).join(' · ');
    ph.innerHTML = `Loading 3 AI models (~36 MB, cached after first visit)…<br><small>${parts}</small>`;
  };
  const [g, w1, w2] = await Promise.all([
    fetchModel('general.onnx', p => { progress.general = p; report(); }),
    fetchModel('weapons.onnx', p => { progress.weapons = p; report(); }),
    fetchModel('weapons2.onnx', p => { progress.weapons2 = p; report(); }),
  ]);
  const general = await ort.InferenceSession.create(g, opts);
  const weapons = [
    await ort.InferenceSession.create(w1, opts),
    await ort.InferenceSession.create(w2, opts),
  ];
  sessions = { general, weapons };
  ph.textContent = 'Models ready. Start the camera or try the sample image.';
  $('btnCam').disabled = false;
  $('btnSample').disabled = false;
  const backend = navigator.gpu
    ? 'webgpu'
    : `wasm ×${crossOriginIsolated ? ort.env.wasm.numThreads : 1} threads`;
  $('perf').textContent = `backend: ${backend}`;
}

// -------------------------------------------------------------- inference --
const prep = document.createElement('canvas');
prep.width = MODEL_SIZE; prep.height = MODEL_SIZE;
const prepCtx = prep.getContext('2d', { willReadFrequently: true });

function letterbox(source, sw, sh) {
  const scale = Math.min(MODEL_SIZE / sw, MODEL_SIZE / sh);
  const nw = Math.round(sw * scale), nh = Math.round(sh * scale);
  const dx = (MODEL_SIZE - nw) / 2, dy = (MODEL_SIZE - nh) / 2;
  prepCtx.fillStyle = '#727272';
  prepCtx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  prepCtx.drawImage(source, 0, 0, sw, sh, dx, dy, nw, nh);
  const { data } = prepCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
  const n = MODEL_SIZE * MODEL_SIZE;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i] = data[i * 4] / 255;
    input[n + i] = data[i * 4 + 1] / 255;
    input[2 * n + i] = data[i * 4 + 2] / 255;
  }
  return { input, scale, dx, dy };
}

function decode(output, names, conf, lb, fromWeaponModel) {
  // output: Float32Array [1, 4+nc, 8400]
  const nc = names.length, anchors = 8400;
  const boxes = [];
  for (let a = 0; a < anchors; a++) {
    let best = 0, bestCls = -1;
    for (let c = 0; c < nc; c++) {
      const s = output[(4 + c) * anchors + a];
      if (s > best) { best = s; bestCls = c; }
    }
    if (best < conf) continue;
    const cx = output[a], cy = output[anchors + a];
    const w = output[2 * anchors + a], h = output[3 * anchors + a];
    const label = names[bestCls];
    const kind = kindFor(label, fromWeaponModel);
    boxes.push({
      label, kind, confidence: best,
      x1: (cx - w / 2 - lb.dx) / lb.scale, y1: (cy - h / 2 - lb.dy) / lb.scale,
      x2: (cx + w / 2 - lb.dx) / lb.scale, y2: (cy + h / 2 - lb.dy) / lb.scale,
    });
  }
  return nms(boxes);
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}

function nms(boxes) {
  boxes.sort((a, b) => b.confidence - a.confidence);
  const keep = [];
  for (const box of boxes) {
    if (keep.every(k => k.label !== box.label || iou(k, box) < NMS_IOU)) keep.push(box);
  }
  return keep;
}

function mergeCandidates(perModel) {
  // Collapse same-label overlapping boxes from different weapons models into
  // one detection that remembers how many models agreed.
  const merged = [];
  perModel.forEach((candidates, modelIdx) => {
    for (const cand of candidates) {
      const match = merged.find(m => m.label === cand.label
                                     && iou(m, cand) > ENSEMBLE_IOU);
      if (match) {
        match.models.add(modelIdx);
        if (cand.confidence > match.confidence) {
          match.confidence = cand.confidence;
          match.x1 = cand.x1; match.y1 = cand.y1;
          match.x2 = cand.x2; match.y2 = cand.y2;
        }
      } else {
        cand.models = new Set([modelIdx]);
        merged.push(cand);
      }
    }
  });
  return merged;
}

function fuseWeapons(candidates, general, sw, sh) {
  const frameArea = sw * sh;
  const persons = general.filter(d => d.kind === 'person');
  const cocoKnives = general.filter(d => d.label === 'knife');
  const distractors = general.filter(
    d => DISTRACTORS.has(d.label) && d.confidence >= DISTRACTOR_MIN_CONF);

  const accepted = [];
  for (const cand of candidates) {
    const areaFrac = (cand.x2 - cand.x1) * (cand.y2 - cand.y1) / frameArea;
    // Huge boxes are misfires — real handheld weapons are small in frame
    if (areaFrac > WEAPON_MAX_AREA_FRAC) continue;
    // A "weapon" box that coincides with a person box IS the person
    if (persons.some(p => iou(cand, p) > WEAPON_PERSON_IOU_VETO)) continue;
    // Confident everyday object in the same spot (laptop, phone, ...)
    if (distractors.some(d => iou(cand, d) > DISTRACTOR_IOU
                              && d.confidence > 0.8 * cand.confidence)) continue;

    // Tiered acceptance: use the lowest bar this candidate qualifies for
    const bars = [['high_conf', WEAPON_CONF_ALONE]];
    if (persons.some(p => boxesNear(cand, p)))
      bars.push(['near_person', WEAPON_CONF_NEAR_PERSON]);
    if ((cand.models || new Set()).size >= 2)
      bars.push(['ensemble_agree', WEAPON_CONF_ENSEMBLE]);
    if (cand.label === 'knife' && cocoKnives.some(k => iou(cand, k) > 0.3))
      bars.push(['agrees_coco', WEAPON_CONF_AGREE]);

    const [basis, bar] = bars.reduce((a, b) => a[1] <= b[1] ? a : b);
    if (cand.confidence >= bar) { cand.basis = basis; accepted.push(cand); }
  }
  return accepted;
}

async function detect(source, sw, sh) {
  const start = performance.now();
  const lb = letterbox(source, sw, sh);
  const tensor = new ort.Tensor('float32', lb.input, [1, 3, MODEL_SIZE, MODEL_SIZE]);

  const gOut = await sessions.general.run({ [sessions.general.inputNames[0]]: tensor });
  const general = decode(gOut[sessions.general.outputNames[0]].data, COCO, GENERAL_CONF, lb, false);

  const perModel = [];
  for (const session of sessions.weapons) {
    const out = await session.run({ [session.inputNames[0]]: tensor });
    perModel.push(decode(out[session.outputNames[0]].data, WEAPON_NAMES,
                         WEAPON_CANDIDATE_CONF, lb, true));
  }
  const weapons = fuseWeapons(mergeCandidates(perModel), general, sw, sh);

  const dets = [...general.filter(d => d.kind !== 'other'), ...weapons];
  return { dets, ms: performance.now() - start };
}

// ---------------------------------------------------------------- tracker --
let tracks = [], nextTrackId = 1;

function updateTracks(persons) {
  const now = performance.now() / 1000;
  const unmatched = [...persons];
  for (const t of tracks) {
    let best = null, bestDist = 120;
    for (const d of unmatched) {
      const cx = (d.x1 + d.x2) / 2, cy = (d.y1 + d.y2) / 2;
      const dist = Math.hypot(cx - t.cx, cy - t.cy);
      if (dist < bestDist) { best = d; bestDist = dist; }
    }
    if (best) {
      unmatched.splice(unmatched.indexOf(best), 1);
      t.cx = (best.x1 + best.x2) / 2; t.cy = (best.y1 + best.y2) / 2;
      t.box = best; t.lastSeen = now;
      t.positions.push([now, t.cx, t.cy]);
      if (t.positions.length > 150) t.positions.shift();
      best.trackId = t.id;
    }
  }
  for (const d of unmatched) {
    const cx = (d.x1 + d.x2) / 2, cy = (d.y1 + d.y2) / 2;
    const t = { id: nextTrackId++, cx, cy, box: d, firstSeen: now, lastSeen: now,
                positions: [[now, cx, cy]], nearVehicleSince: null };
    d.trackId = t.id;
    tracks.push(t);
  }
  tracks = tracks.filter(t => now - t.lastSeen < 2);
  return tracks;
}

function travelRadius(t) {
  if (t.positions.length < 2) return 0;
  let sx = 0, sy = 0;
  for (const [, x, y] of t.positions) { sx += x; sy += y; }
  const mx = sx / t.positions.length, my = sy / t.positions.length;
  return Math.max(...t.positions.map(([, x, y]) => Math.hypot(x - mx, y - my)));
}

function trackSpeed(t) {
  // Mean speed in px/s over roughly the last second of history
  if (t.positions.length < 2) return 0;
  const tNow = t.positions[t.positions.length - 1][0];
  const recent = t.positions.filter(p => tNow - p[0] <= 1.2);
  if (recent.length < 2) return 0;
  const dt = recent[recent.length - 1][0] - recent[0][0];
  if (dt <= 0) return 0;
  return Math.hypot(recent[recent.length - 1][1] - recent[0][1],
                    recent[recent.length - 1][2] - recent[0][2]) / dt;
}

// ----------------------------------------------------------- threat rules --
let weaponWindow = [], weaponFirstSeen = null, fightWindow = [];

function boxesNear(a, b, margin = 40) {
  return !(a.x2 + margin < b.x1 || b.x2 + margin < a.x1 ||
           a.y2 + margin < b.y1 || b.y2 + margin < a.y1);
}

function classify(dets, w, h, isStatic = false) {
  const threats = [];
  const now = performance.now() / 1000;
  const weapons = dets.filter(d => d.kind === 'weapon');
  const vehicles = dets.filter(d => d.kind === 'vehicle');
  const persons = dets.filter(d => d.kind === 'person');
  const dangers = dets.filter(d => d.kind === 'danger');

  // Windowed vote: a weapon must show up in most recent frames before it
  // counts, so single-frame flickers never raise an alert. A single
  // uploaded image has no history, so it bypasses the vote.
  let confirmed;
  if (isStatic) {
    confirmed = weapons.length > 0;
    weaponFirstSeen = null;   // persistence is meaningless for one image
  } else {
    weaponWindow.push(weapons.length > 0);
    if (weaponWindow.length > WEAPON_WINDOW) weaponWindow.shift();
    const votes = weaponWindow.filter(Boolean).length;
    confirmed = votes >= WEAPON_VOTES;
    if (confirmed) {
      if (weaponFirstSeen === null) weaponFirstSeen = now;
    } else if (votes === 0) weaponFirstSeen = null;
  }

  if (confirmed && weapons.length) {
    const top = weapons.reduce((a, b) => a.confidence > b.confidence ? a : b);
    const persisted = weaponFirstSeen === null ? 0 : now - weaponFirstSeen;
    if (persisted >= WEAPON_CRITICAL_SECONDS) {
      threats.push({ level: 'CRITICAL', kind: 'weapon_persistent',
        message: `${top.label.toUpperCase()} persisting in scene for ${persisted.toFixed(0)}s — possible active threat` });
    } else {
      threats.push({ level: 'HIGH', kind: 'weapon',
        message: `${top.label.toUpperCase()} detected (${(top.confidence * 100).toFixed(0)}% confidence)` });
    }
  }

  // Altercation: two people moving fast in close quarters
  let fightingPair = null;
  if (!isStatic) {
    for (let i = 0; i < tracks.length; i++) {
      for (let j = i + 1; j < tracks.length; j++) {
        const a = tracks[i], b = tracks[j];
        if (boxesNear(a.box, b.box, 20)
            && trackSpeed(a) > FIGHT_SPEED_PX && trackSpeed(b) > FIGHT_SPEED_PX) {
          fightingPair = [a, b];
        }
      }
    }
    fightWindow.push(fightingPair !== null);
    if (fightWindow.length > FIGHT_WINDOW) fightWindow.shift();
    if (fightingPair && fightWindow.filter(Boolean).length >= FIGHT_VOTES) {
      threats.push({ level: 'HIGH', kind: 'altercation',
        message: `Rapid close-quarters movement between person #${fightingPair[0].id} and #${fightingPair[1].id} — possible altercation` });
    }
  }

  // Dangerous object carried by a person
  for (const d of dangers) {
    if (persons.some(p => boxesNear(d, p))) {
      threats.push({ level: 'MEDIUM', kind: 'suspicious_object',
        message: `${d.label[0].toUpperCase() + d.label.slice(1)} (${(d.confidence * 100).toFixed(0)}%) near a person` });
    }
  }

  for (const t of tracks) {
    if (vehicles.some(v => boxesNear(t.box, v))) {
      if (t.nearVehicleSince === null) t.nearVehicleSince = now;
      else if (now - t.nearVehicleSince >= VEHICLE_LURK_SECONDS) {
        threats.push({ level: 'MEDIUM', kind: 'vehicle_lurking',
          message: `Person #${t.id} lingering near a vehicle for ${(now - t.nearVehicleSince).toFixed(0)}s` });
      }
    } else t.nearVehicleSince = null;
  }

  // Dwell-based zone rule: someone must STAY in the zone (5 s), not just
  // walk past it. Static images alert immediately (no dwell history).
  if ($('chkZone').checked) {
    if (isStatic) {
      for (const p of persons) {
        if ((p.x1 + p.x2) / 2 >= 0.6 * w) {
          threats.push({ level: 'MEDIUM', kind: 'trespassing',
            message: `Person ${p.trackId ? '#' + p.trackId : ''} inside restricted zone` });
        }
      }
    } else {
      for (const t of tracks) {
        const cx = (t.box.x1 + t.box.x2) / 2;
        if (cx >= 0.6 * w) {
          if (t.zoneSince == null) t.zoneSince = now;
          else if (now - t.zoneSince >= ZONE_DWELL_SECONDS) {
            threats.push({ level: 'MEDIUM', kind: 'trespassing',
              message: `Person #${t.id} inside restricted zone for ${(now - t.zoneSince).toFixed(0)}s` });
          }
        } else t.zoneSince = null;
      }
    }
  }

  for (const t of tracks) {
    const age = t.lastSeen - t.firstSeen;
    if (age >= LOITER_SECONDS && travelRadius(t) < LOITER_RADIUS) {
      threats.push({ level: 'LOW', kind: 'loitering',
        message: `Person #${t.id} loitering for ${age.toFixed(0)}s` });
    }
  }
  return threats;
}

const highestLevel = threats => threats.length
  ? threats.reduce((a, b) => LEVELS.indexOf(a.level) > LEVELS.indexOf(b.level) ? a : b).level
  : null;

// ------------------------------------------------------------ alert engine --
const cooldowns = {};
let alertHistory = [];
try { alertHistory = JSON.parse(localStorage.getItem('alerts') || '[]'); } catch {}

function fireAlerts(threats) {
  const now = performance.now() / 1000;
  for (const t of threats) {
    const key = t.level + '|' + t.kind;
    if (cooldowns[key] && now - cooldowns[key] < ALERT_COOLDOWN_SECONDS) continue;
    cooldowns[key] = now;

    const thumb = document.createElement('canvas');
    const scale = 220 / canvas.width;
    thumb.width = 220; thumb.height = Math.round(canvas.height * scale);
    thumb.getContext('2d').drawImage(canvas, 0, 0, thumb.width, thumb.height);

    const alert = {
      ts: new Date().toISOString().slice(0, 19),
      level: t.level, kind: t.kind, message: t.message,
      snapshot: thumb.toDataURL('image/jpeg', 0.6),
    };
    alertHistory.unshift(alert);
    alertHistory = alertHistory.slice(0, 40);
    try { localStorage.setItem('alerts', JSON.stringify(alertHistory)); }
    catch { /* quota exceeded — keep in-memory only */ }

    addAlertToFeed(alert, true);
    tgSendAlert(alert);
    if (t.level === 'HIGH' || t.level === 'CRITICAL') beep(t.level === 'CRITICAL');
  }
}

function addAlertToFeed(a, prepend) {
  const feed = $('feed');
  const emptyEl = feed.querySelector('.empty');
  if (emptyEl) emptyEl.remove();
  const el = document.createElement('div');
  el.className = 'alert-item';
  const fname = `alert_${a.ts.replace(/[:T-]/g, '')}_${a.level}.jpg`;
  el.innerHTML = `
    <span class="badge ${a.level}">${a.level}</span>
    <div class="msg">${a.message}<div class="ts">${a.ts.replace('T', ' ')}</div></div>
    ${a.snapshot ? `<div style="display:flex;flex-direction:column;gap:3px;align-items:center">
       <img src="${a.snapshot}" title="alert snapshot">
       <a href="${a.snapshot}" download="${fname}"
          style="font-size:11px;color:var(--muted)">⬇ save</a></div>` : ''}`;
  if (a.snapshot) el.querySelector('img').onclick = e => window.open(e.target.src);
  prepend ? feed.prepend(el) : feed.append(el);
  while (feed.children.length > 60) feed.lastChild.remove();
}

function beep(critical) {
  if (!$('chkSound').checked) return;
  const now = Date.now();
  if (now - lastBeep < 1200) return;
  lastBeep = now;
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const osc = audioCtx.createOscillator(), gain = audioCtx.createGain();
  osc.frequency.value = critical ? 1400 : 900;
  gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.6);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(); osc.stop(audioCtx.currentTime + 0.6);
}

// -------------------------------------------------------------- rendering --
function render(source, sw, sh, dets, threats, ms) {
  canvas.width = sw; canvas.height = sh;
  ctx.drawImage(source, 0, 0, sw, sh);

  if ($('chkZone').checked) {
    ctx.strokeStyle = '#00c8ff'; ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(0.6 * sw, 0, 0.4 * sw, sh);
    ctx.setLineDash([]);
    ctx.fillStyle = '#00c8ff'; ctx.font = '600 14px system-ui';
    ctx.fillText('RESTRICTED', 0.6 * sw + 8, 20);
  }

  for (const d of dets) {
    const color = KIND_COLORS[d.kind] || '#999';
    ctx.lineWidth = d.kind === 'weapon' ? 4 : 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    let label = d.label;
    if (d.trackId) label += ` #${d.trackId}`;
    label += ` ${(d.confidence * 100).toFixed(0)}%`;
    ctx.font = '600 13px system-ui';
    const w = ctx.measureText(label).width + 10;
    ctx.fillStyle = color;
    ctx.fillRect(d.x1, Math.max(0, d.y1 - 20), w, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, d.x1 + 5, Math.max(14, d.y1 - 6));
  }

  const level = highestLevel(threats);
  const banner = $('levelBanner');
  banner.className = 'level-banner' + (level ? ' ' + level : '');
  banner.textContent = level ? 'THREAT: ' + level : 'ALL CLEAR';
  if (level) {
    ctx.fillStyle = LEVEL_COLORS[level];
    ctx.fillRect(0, 0, sw, 30);
    ctx.fillStyle = '#fff'; ctx.font = '700 16px system-ui';
    ctx.fillText('THREAT LEVEL: ' + level, 10, 21);
  }

  const counts = {};
  for (const d of dets) counts[d.kind] = (counts[d.kind] || 0) + 1;
  $('stPersons').textContent = counts.person || 0;
  $('stVehicles').textContent = counts.vehicle || 0;
  $('stWeapons').textContent = counts.weapon || 0;
  $('stPackages').textContent = counts.package || 0;
  $('perf').textContent = `inference ${ms.toFixed(0)} ms · ${(1000 / ms).toFixed(1)} fps max`;
}

// ------------------------------------------------------------------ loops --
async function processFrame(source, sw, sh, isStatic = false) {
  const { dets, ms } = await detect(source, sw, sh);
  updateTracks(dets.filter(d => d.kind === 'person'));
  const threats = classify(dets, sw, sh, isStatic);
  render(source, sw, sh, dets, threats, ms);
  fireAlerts(threats);
  return { dets, threats, ms };
}

async function cameraLoop() {
  while (running) {
    if (video.readyState >= 2) {
      try { await processFrame(video, video.videoWidth, video.videoHeight); }
      catch (e) { console.error(e); }
    }
    await new Promise(r => setTimeout(r, 30));
  }
}

$('btnCam').onclick = async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 } });
  } catch (e) {
    $('error').textContent = 'Camera unavailable: ' + e.message;
    return;
  }
  video.srcObject = stream;
  await video.play();
  $('placeholder').hidden = true;
  canvas.hidden = false;
  $('statusDot').classList.add('on');
  $('btnCam').disabled = true;
  $('btnStop').disabled = false;
  running = true;
  tracks = []; weaponWindow = []; weaponFirstSeen = null; fightWindow = [];
  cameraLoop();
};

$('btnStop').onclick = () => {
  running = false;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $('statusDot').classList.remove('on');
  $('btnCam').disabled = false;
  $('btnStop').disabled = true;
};

async function detectImage(img) {
  $('btnStop').onclick();
  $('placeholder').hidden = true;
  canvas.hidden = false;
  tracks = []; weaponWindow = []; weaponFirstSeen = null; fightWindow = [];
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const result = await processFrame(img, w, h, true);
  return result;
}

$('btnSample').onclick = async () => {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = () => {   // not bundled in the deploy — pull from repo CDN
      img.onerror = rej;
      img.src = GITHUB_MODEL_BASE.replace('/models/', '/') + 'sample.jpg';
    };
    img.src = 'sample.jpg';
  });
  const r = await detectImage(img);
  window.__lastResult = r;   // for automated tests
};

$('fileInput').onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  await new Promise((res, rej) => {
    img.onload = res; img.onerror = rej; img.src = URL.createObjectURL(file);
  });
  await detectImage(img);
};

// --------------------------------------------------------------- telegram --
// Browser-side Telegram alerts: the bot token and chat id live only in this
// browser's localStorage and calls go straight to api.telegram.org — no
// server involved. The full verdict-button loop lives in the Python edition.
let tgCfg = { token: '', chat: '', min: 'HIGH' };
try { Object.assign(tgCfg, JSON.parse(localStorage.getItem('tg_cfg') || '{}')); } catch {}

function tgStatus(text, ok) {
  const el = $('tgStatus');
  el.textContent = text;
  el.style.color = ok ? 'var(--ok)' : 'var(--muted)';
}

function tgRefreshUi() {
  $('tgToken').value = tgCfg.token;
  $('tgChat').value = tgCfg.chat;
  $('tgLevel').value = tgCfg.min;
  tgStatus(tgCfg.token && tgCfg.chat
    ? `Configured — sends ${tgCfg.min}+ alerts with snapshots`
    : (tgCfg.token ? 'Token saved — detect or enter your chat id'
                   : 'Not configured'), !!(tgCfg.token && tgCfg.chat));
}

async function tgApi(method, body) {
  const res = await fetch(
    `https://api.telegram.org/bot${tgCfg.token}/${method}`,
    { method: 'POST', body });
  const data = await res.json();
  if (!data.ok) throw new Error(data.description || 'Telegram error');
  return data.result;
}

function dataUrlToBlob(dataUrl) {
  const [head, b64] = dataUrl.split(',');
  const mime = head.match(/data:(.*?);/)[1];
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

function tgSendAlert(alert) {
  if (!tgCfg.token || !tgCfg.chat) return;
  if (LEVELS.indexOf(alert.level) < LEVELS.indexOf(tgCfg.min)) return;
  try {
    const form = new FormData();
    form.append('chat_id', tgCfg.chat);
    form.append('caption',
      `${alert.level}: ${alert.message}\n${alert.ts.replace('T', ' ')}`);
    form.append('photo', dataUrlToBlob(alert.snapshot), 'alert.jpg');
    tgApi('sendPhoto', form)
      .catch(e => tgStatus('Send failed: ' + e.message, false));
  } catch (e) { console.error('telegram', e); }
}

$('tgSave').onclick = () => {
  tgCfg = { token: $('tgToken').value.trim(), chat: $('tgChat').value.trim(),
            min: $('tgLevel').value };
  localStorage.setItem('tg_cfg', JSON.stringify(tgCfg));
  tgRefreshUi();
};

$('tgDetect').onclick = async () => {
  tgCfg.token = $('tgToken').value.trim();
  if (!tgCfg.token) { tgStatus('Enter the bot token first', false); return; }
  tgStatus('Looking for your chat…', false);
  try {
    const updates = await tgApi('getUpdates',
      new URLSearchParams({ timeout: '0' }));
    const msg = [...updates].reverse().find(u => u.message);
    if (!msg) throw new Error(
      'no messages — open your bot in Telegram, press Start, then retry');
    $('tgChat').value = String(msg.message.chat.id);
    $('tgSave').onclick();
  } catch (e) { tgStatus('Detect failed: ' + e.message, false); }
};

$('tgTest').onclick = async () => {
  tgCfg = { token: $('tgToken').value.trim(), chat: $('tgChat').value.trim(),
            min: $('tgLevel').value };
  if (!tgCfg.token || !tgCfg.chat) {
    tgStatus('Need token and chat id first', false); return;
  }
  try {
    const form = new FormData();
    form.append('chat_id', tgCfg.chat);
    form.append('text',
      '🔔 Test from AI CCTV browser demo — alerts are working.');
    await tgApi('sendMessage', form);
    tgStatus('Test sent — check your Telegram', true);
  } catch (e) { tgStatus('Test failed: ' + e.message, false); }
};

// ------------------------------------------------------------------- init --
for (const a of alertHistory) addAlertToFeed(a, false);
tgRefreshUi();
loadModels().catch(e => {
  $('placeholder').textContent = 'Failed to load models: ' + e.message;
});

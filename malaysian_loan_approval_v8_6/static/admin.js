// Malaysian Loan Admin Console — compact API-connected dashboard
const $ = (id) => document.getElementById(id);
const plotConfig = { displayModeBar: false, responsive: true };
const ADMIN_TOKEN = new URLSearchParams(window.location.search).get('token') || localStorage.getItem('ADMIN_TOKEN') || '';
if (ADMIN_TOKEN) localStorage.setItem('ADMIN_TOKEN', ADMIN_TOKEN);
function authHeaders(extra = {}) { return ADMIN_TOKEN ? { ...extra, 'X-Admin-Token': ADMIN_TOKEN } : extra; }
function tokenQuery() { return ADMIN_TOKEN ? `?token=${encodeURIComponent(ADMIN_TOKEN)}` : ''; }
function esc(v) { return String(v ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function fmt(v, digits = 2) { const n = Number(v); return Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : (v == null ? '—' : esc(v)); }
function fmtFixed(v, digits = 5) { const n = Number(v); return Number.isFinite(n) ? n.toFixed(digits) : (v == null ? '—' : esc(v)); }
function cleanSource(v) { return String(v || '').replace(/\s*\(healed\)/ig, '').replace('Holdout-excluded GA baseline', 'Holdout-excluded baseline').replace('Boot-time GA baseline (full data)', 'Boot baseline'); }
function metricCard(k, v, d, extra='') { return `<article class="metric-card ${extra}"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="d">${esc(d)}</div></article>`; }

const sectionButtons = Array.from(document.querySelectorAll('.admin-nav'));
const sections = Array.from(document.querySelectorAll('.admin-section'));
let gaLoaded = false;
let mfLoaded = false;
let currentMembershipView = 'all';
let pollTimer = null;
let lastState = null;
const openRunDetails = new Set();
let hasLoadedStateOnce = false;
let lastKnownRunKey = null;

function showAdminSection(name) {
  sectionButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.adminSection === name));
  sections.forEach(section => section.classList.toggle('active', section.id === `admin-section-${name}`));
  document.querySelector('.admin-shell')?.scrollTo({ top: 0, behavior: 'smooth' });
  if (name === 'ga' && !gaLoaded) loadGA();
  if (name === 'membership' && !mfLoaded) loadMembership('all', true);
  setTimeout(resizePlots, 220);
}
sectionButtons.forEach(btn => btn.addEventListener('click', () => showAdminSection(btn.dataset.adminSection)));

function resizePlots() {
  const plotEls = ['gaChart', 'membershipChart']
    .map(id => $(id))
    .filter(Boolean)
    .concat(Array.from(document.querySelectorAll('.mf-card-chart')));
  plotEls.forEach(el => {
    if (el && window.Plotly) { try { Plotly.Plots.resize(el); } catch (_) {} }
  });
}
window.addEventListener('resize', resizePlots);

function showToast(message, type = 'info') {
  const host = $('toastHost');
  if (!host) return;
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  host.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 220); }, 4200);
}
function resetCsvPicker() {
  selectedCsvFile = null;
  const input = $('csvFile');
  if (input) input.value = '';
  const fileName = $('fileName');
  if (fileName) fileName.textContent = 'No file selected';
  $('uploadZone')?.classList.remove('has-file', 'dragging', 'busy');
}
function updateAgentProgress(state, message = '') {
  const box = $('agentProgress');
  if (!box) return;
  const s = String(state || 'idle').toLowerCase();
  box.classList.toggle('is-running', s === 'running');
  box.classList.toggle('is-done', s === 'done');
  box.classList.toggle('is-error', s === 'error' || s === 'failed');
  const text = $('agentProgressText');
  if (text) {
    if (s === 'running') text.textContent = message || 'Running GA worker…';
    else if (s === 'done') text.textContent = message || 'Latest retraining run completed.';
    else if (s === 'error' || s === 'failed') text.textContent = message || 'Retraining failed. Check logs.';
    else text.textContent = 'Idle · waiting for clean rows';
  }
  const steps = Array.from(box.querySelectorAll('[data-step]'));
  steps.forEach((el, idx) => {
    el.classList.remove('active', 'done', 'idle');
    if (s === 'running') {
      if (idx < 2) el.classList.add('done');
      if (idx === 2 || idx === 3) el.classList.add('active');
    } else if (s === 'done') {
      el.classList.add('done');
    } else {
      el.classList.add('idle');
    }
  });
  const pill = $('cloudTaskPill');
  pill?.classList.toggle('pulse', s === 'running');
}
function maybeToastLatestRun(d, running) {
  const runs = d.runs || [];
  const latest = runs[0];
  const key = latest ? `${latest.run_id || ''}:${latest.status || ''}:${latest.version ?? ''}` : null;
  if (!hasLoadedStateOnce) { lastKnownRunKey = key; hasLoadedStateOnce = true; return; }
  if (!running && key && key !== lastKnownRunKey) {
    const status = String(latest.status || '').toLowerCase();
    if (status === 'promoted') showToast(`Model v${latest.version} promoted · blended MSE improved.`, 'success');
    else if (status === 'skipped') showToast('Candidate skipped · active model kept.', 'warning');
    else if (status === 'failed') showToast('Retraining failed · pending rows kept for retry.', 'error');
    lastKnownRunKey = key;
  }
}

// ---------------------------------------------------------------------------
// Upload UX helpers
// ---------------------------------------------------------------------------
let selectedCsvFile = null;
function isCsvFile(file) { return !!file && (file.name || '').toLowerCase().endsWith('.csv'); }
function updateSelectedCsv(file) {
  selectedCsvFile = file || null;
  const fileName = $('fileName');
  const zone = $('uploadZone');
  if (fileName) fileName.textContent = selectedCsvFile ? selectedCsvFile.name : 'No file selected';
  zone?.classList.toggle('has-file', Boolean(selectedCsvFile));
  if (selectedCsvFile && !isCsvFile(selectedCsvFile)) {
    setUploadMessage('Please choose a .csv file.', 'error');
  } else if (selectedCsvFile) {
    setUploadMessage('CSV selected. Click Upload & Validate to continue.', 'info');
  }
}
function setUploadMessage(text, type = 'info', loading = false) {
  const el = $('uploadMsg');
  if (!el) return;
  el.className = `helper-text upload-message ${type} ${loading ? 'loading' : ''}`;
  el.innerHTML = loading ? `<span class="spinner"></span><span>${esc(text)}</span>` : esc(text);
}
function setButtonBusy(btn, busy, label) {
  if (!btn) return;
  if (!btn.dataset.originalHtml) btn.dataset.originalHtml = btn.innerHTML;
  btn.disabled = Boolean(busy);
  btn.classList.toggle('is-loading', Boolean(busy));
  btn.innerHTML = busy ? `<span class="spinner light"></span>${esc(label || 'Working…')}` : btn.dataset.originalHtml;
}
function setUploadBusy(busy) {
  setButtonBusy($('uploadBtn'), busy, 'Validating…');
  $('uploadZone')?.classList.toggle('busy', Boolean(busy));
}

$('csvFile')?.addEventListener('change', () => updateSelectedCsv($('csvFile').files?.[0] || null));
const uploadZone = $('uploadZone');
if (uploadZone) {
  const stop = e => { e.preventDefault(); e.stopPropagation(); };
  // Prevent the browser from opening a dragged CSV if the user misses the drop box.
  ['dragover', 'drop'].forEach(evt => document.addEventListener(evt, e => {
    if (e.dataTransfer?.types?.includes('Files')) e.preventDefault();
  }));
  ['dragenter', 'dragover'].forEach(evt => uploadZone.addEventListener(evt, e => {
    stop(e);
    uploadZone.classList.add('dragging');
    setUploadMessage('Drop the CSV file here.', 'info');
  }));
  ['dragleave', 'dragend'].forEach(evt => uploadZone.addEventListener(evt, e => {
    stop(e);
    if (!uploadZone.contains(e.relatedTarget)) uploadZone.classList.remove('dragging');
  }));
  uploadZone.addEventListener('drop', e => {
    stop(e);
    uploadZone.classList.remove('dragging');
    const file = Array.from(e.dataTransfer?.files || []).find(isCsvFile) || e.dataTransfer?.files?.[0];
    if (!file) { setUploadMessage('Drop a .csv file to upload.', 'error'); return; }
    updateSelectedCsv(file);
  });
}

async function loadState() {
  try {
    const r = await fetch('/api/admin/state', { headers: authHeaders() });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'Failed to load');
    lastState = d;
    renderState(d);
  } catch (e) {
    if ($('statusMsg')) $('statusMsg').textContent = 'Could not load admin state. Open /admin?token=YOUR_TOKEN if ADMIN_TOKEN is enabled.';
  }
}

function renderState(d) {
  const st = d.status || {};
  const state = String(st.state || 'idle').toLowerCase();
  const f = d.freshness || {};
  const pending = d.pending || [];
  const review = d.review || [];
  const versions = d.versions || [];
  const runs = d.runs || [];
  const ready = Boolean(f.ready);
  const running = state === 'running';

  const stateLabel = state.charAt(0).toUpperCase() + state.slice(1);
  if ($('statusMsg')) $('statusMsg').textContent = st.message || '—';
  if ($('statusStateText')) $('statusStateText').textContent = stateLabel;
  if ($('currentJobText')) $('currentJobText').textContent = running ? 'Retraining Agent / GA worker' : 'None';
  if ($('lastUpdated')) $('lastUpdated').textContent = st.updated || '—';
  const pill = $('statusPill');
  if (pill) { pill.textContent = state.toUpperCase(); pill.className = 'process-status ' + (running ? 'running' : state === 'done' ? 'done' : state === 'error' ? 'error' : ''); }

  if ($('adminMetrics')) {
    const currentBatch = Number(d.current_batch?.row_count ?? st.batch_rows ?? 0);
    const queuedRows = Number(d.pending_total ?? pending.length);
    $('adminMetrics').innerHTML = [
      metricCard('Retrain Trigger', `${f.new_rows ?? 0} / ${f.threshold ?? 0}`, running ? 'queued for next run' : (ready ? 'ready to retrain' : 'collecting')),
      metricCard('Active Version', d.active_version ? `v${d.active_version}` : 'v0 boot GA', 'serving model'),
      metricCard('Current Run Batch', currentBatch ? fmt(currentBatch, 0) : '—', running ? 'locked for current GA run' : 'no active batch'),
      metricCard('Queued Clean Rows', queuedRows, queuedRows > pending.length ? `showing first ${pending.length}` : (running ? 'waiting for next retrain' : 'waiting for retrain')),
      metricCard('Review Queue', d.review_total ?? review.length, (d.review_total ?? review.length) > review.length ? `showing first ${review.length}` : 'needs manual review'),
    ].join('');
  }

  if ($('retrainBtn')) { $('retrainBtn').disabled = running || !ready; $('retrainBtn').title = ready ? 'Start retraining' : `Requires ${f.threshold || 30} clean rows`; }
  if ($('forceRetrainBtn')) $('forceRetrainBtn').disabled = running;

  updateAgentProgress(state, st.message || '');
  maybeToastLatestRun(d, running);
  if ($('retrainBtn') && running) {
    $('retrainBtn').innerHTML = '<span class="spinner light"></span>Retraining…';
  }
  if ($('forceRetrainBtn') && running) {
    $('forceRetrainBtn').innerHTML = '<span class="material-symbols-outlined">lock</span>Locked';
  }
  if (!running) {
    if ($('retrainBtn')) $('retrainBtn').innerHTML = '<span class="material-symbols-outlined">play_arrow</span>Retrain now';
    if ($('forceRetrainBtn')) $('forceRetrainBtn').innerHTML = '<span class="material-symbols-outlined">warning</span>Force retrain';
  }

  renderPending(pending, d.pending_total, d.pending_preview_limit);
  renderReview(review, d.review_total, d.review_preview_limit);
  renderVersions(d, versions);
  renderRuns(runs);
  if ($('changelog')) $('changelog').textContent = (d.changelog || []).join('\n') || 'Waiting for retraining events...';

  // If a retrain was started by upload/cron or the user refreshed while it was
  // already running, keep polling automatically. This prevents the UI from
  // appearing stuck at Running until the user manually refreshes.
  if (running) poll();
}

function renderPending(pending, total = null, previewLimit = null) {
  const body = document.querySelector('#pendingTable tbody');
  const note = $('pendingPreviewNote');
  if (note) {
    const n = Number(total ?? pending.length);
    note.textContent = n > pending.length ? `Showing ${pending.length} of ${n.toLocaleString()} queued clean rows.` : '';
  }
  if (!body) return;
  body.innerHTML = pending.slice(0, 80).map(r => `
    <tr><td>${fmt(r.person_income, 0)}</td><td>${esc(r.credit_score)}</td><td>${fmt(r.loan_percent_income, 2)}</td><td>${esc(r.person_emp_exp)}</td><td>${esc(r.previous_loan_defaults_on_file)}</td><td>${esc(r.loan_status)}</td></tr>
  `).join('') || '<tr><td colspan="6" class="empty">No records waiting for retraining.</td></tr>';
}

function reviewValue(row, key, fallback = '—') {
  return row && row[key] !== undefined && row[key] !== null && row[key] !== '' ? row[key] : fallback;
}
function renderReview(review, total = null, previewLimit = null) {
  const list = $('reviewList');
  const note = $('reviewPreviewNote');
  if (note) {
    const n = Number(total ?? review.length);
    note.textContent = n > review.length ? `Showing ${review.length} of ${n.toLocaleString()} review rows.` : '';
  }
  if (!list) return;
  list.innerHTML = review.slice(0, 80).map((it, i) => {
    const row = it.row || {};
    const reason = (it.reasons || []).join(', ') || 'Review required';
    return `<div class="review-item structured">
      <div class="review-main">
        <div class="review-reason">${esc(reason)}</div>
        <div class="review-grid">
          <span><b>Income</b>${fmt(reviewValue(row, 'person_income'), 0)}</span>
          <span><b>CTOS</b>${esc(reviewValue(row, 'credit_score'))}</span>
          <span><b>Ratio</b>${fmt(reviewValue(row, 'loan_percent_income'), 2)}</span>
          <span><b>Employment</b>${esc(reviewValue(row, 'person_emp_exp'))}</span>
          <span><b>Default</b>${esc(reviewValue(row, 'previous_loan_defaults_on_file'))}</span>
          <span><b>Label</b>${esc(reviewValue(row, 'loan_status'))}</span>
        </div>
      </div>
      <div class="review-actions"><button class="mini ok" data-act="approve" data-i="${i}">Approve</button><button class="mini bad" data-act="discard" data-i="${i}">Discard</button></div>
    </div>`;
  }).join('') || '<div class="empty">No rows require manual review.</div>';
  list.querySelectorAll('button').forEach(b => b.addEventListener('click', async () => {
    setButtonBusy(b, true, b.dataset.act === 'approve' ? 'Approving…' : 'Discarding…');
    try {
      await fetch(`/api/admin/review/${b.dataset.act}`, { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify({ index: Number(b.dataset.i) }) });
      await loadState();
    } finally {
      setButtonBusy(b, false);
    }
  }));
}

function fmtMseValue(v, digits = 5) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(digits) : '—';
}
function fmtRecentMse(v, gate) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  if (String(gate || '').toLowerCase() === 'anchor_only' && n === 0) return '—';
  return n.toFixed(5);
}
function metricDeltaClass(oldValue, newValue) {
  const oldN = Number(oldValue), newN = Number(newValue);
  if (!Number.isFinite(oldN) || !Number.isFinite(newN)) return '';
  if (newN < oldN) return 'delta-good';
  if (newN > oldN) return 'delta-bad';
  return 'delta-neutral';
}
function formatMseChange(oldValue, newValue) {
  const cls = metricDeltaClass(oldValue, newValue);
  const oldTxt = fmtMseValue(oldValue);
  const newTxt = fmtMseValue(newValue);
  const oldN = Number(oldValue), newN = Number(newValue);
  let arrow = '→';
  if (Number.isFinite(oldN) && Number.isFinite(newN)) arrow = newN < oldN ? '↓' : (newN > oldN ? '↑' : '→');
  return `<span class="mse-change ${cls}"><b>${oldTxt}</b><i>${arrow}</i><b>${newTxt}</b></span>`;
}
function compactReason(r) {
  const status = String(r.status || '').toLowerCase();
  const reason = String(r.reason || '');
  if (status === 'promoted') return 'Blended MSE improved; anchor regression stayed within tolerance.';
  if (status === 'skipped') {
    if (String(r.gate_mode || '').toLowerCase() === 'anchor_only') return 'Candidate did not beat the anchor-only gate.';
    return 'Blended MSE did not improve enough.';
  }
  if (status === 'rollback') return 'Admin rollback event.';
  return reason || 'Recorded run.';
}
function renderVersions(d, versions) {
  const activeVersion = d.active_version ? `v${d.active_version}` : 'v0 baseline';
  if ($('currentVersionText')) $('currentVersionText').textContent = activeVersion;
  if ($('currentModelSource')) $('currentModelSource').textContent = d.active_version ? 'Promoted model artifact is serving.' : 'Holdout-excluded baseline is serving.';
  if ($('currentPromotionText')) $('currentPromotionText').textContent = 'Blended gate';
  if ($('rollbackStatusText')) $('rollbackStatusText').textContent = d.active_version ? 'v0 available' : 'Already on v0';

  const body = document.querySelector('#versionTable tbody');
  if (!body) return;
  const rowsHtml = versions.map(v => {
    const isActive = v.version === d.active_version;
    const statusText = isActive ? '<span class="badge-active">serving</span>' : (v.version === 0 ? '<span class="version-status baseline">baseline</span>' : '<span class="version-status archived">archived</span>');
    const promotionScore = v.blended_new_mse ?? v.new_mse;
    const promotionLabel = v.blended_new_mse == null ? (v.version === 0 ? 'anchor baseline' : 'anchor score') : 'blended gate';
    const anchorMse = v.anchor_new_mse ?? v.new_mse;
    const source = v.version === 0 ? 'Holdout-excluded baseline' : (v.source ? cleanSource(v.source) : `Promoted artifact v${v.version}`);
    const trainRows = fmt(v.train_rows || v.total_rows || 0, 0);
    const valRows = v.validation_rows == null ? '—' : fmt(v.validation_rows, 0);
    return `<tr${v.version === 0 ? ' class="baseline-row"' : ''}>`
      + `<td><div class="version-cell"><b>v${esc(v.version)}</b>${isActive ? ' <span class="badge-active">active</span>' : ''}<small>${esc(source)}</small></div></td>`
      + `<td>${statusText}</td>`
      + `<td><div class="metric-stack"><b>${fmtMseValue(promotionScore)}</b><small>${esc(promotionLabel)}</small></div></td>`
      + `<td><div class="metric-stack"><b>${fmtMseValue(anchorMse)}</b><small>fixed anchor</small></div></td>`
      + `<td><div class="metric-stack"><b>${trainRows}</b><small>${valRows} validation</small></div></td>`
      + `<td><button class="mini" data-roll="${esc(v.version)}" ${isActive ? 'disabled' : ''}>Rollback</button> `
      + `<a class="mini" href="/api/admin/download/${esc(v.version)}${tokenQuery()}">Download ZIP</a></td></tr>`;
  }).join('');
  body.innerHTML = rowsHtml || '<tr><td colspan="6" class="empty">No model versions yet.</td></tr>';
  body.querySelectorAll('button[data-roll]:not([disabled])').forEach(b => b.addEventListener('click', async () => {
    if (!confirm(`Rollback to model v${b.dataset.roll}?`)) return;
    await fetch('/api/admin/rollback', { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify({ version: Number(b.dataset.roll) }) });
    loadState();
    loadMembership(currentMembershipView, true);
  }));
}

function runStatusBadge(status) {
  const s = String(status || '').toLowerCase();
  return `<span class="run-status ${s}">${esc(status || 'unknown')}</span>`;
}
function renderRuns(runs) {
  const body = document.querySelector('#historyTable tbody');
  if (!body) return;
  const running = String(lastState?.status?.state || '').toLowerCase() === 'running';
  const runningRow = running ? `<tr class="history-summary-row running-row"><td><div class="history-time"><b>Current run</b><small>waiting for result</small></div></td><td><span class="run-status running">running</span></td><td>—</td><td><span class="gate-pill">working</span></td><td><span class="skeleton-line wide"></span></td><td><span class="skeleton-line"></span></td><td><span class="skeleton-line short"></span></td><td class="history-reason">Retraining Agent is running GA, evaluating candidate, and writing logs.</td><td>—</td></tr>` : '';
  const rowsHtml = runs.slice(0, 80).map((r, i) => {
    const gate = r.gate_mode || '—';
    const anchorOld = r.anchor_old_mse ?? r.old_mse;
    const anchorNew = r.anchor_new_mse ?? r.new_mse;
    const recentOld = r.recent_old_mse;
    const recentNew = r.recent_new_mse;
    const blendedOld = r.blended_old_mse ?? r.old_mse;
    const blendedNew = r.blended_new_mse ?? r.new_mse;
    const anchorReg = r.anchor_regression ?? ((Number(anchorOld) > 0 && Number.isFinite(Number(anchorNew))) ? Number(anchorNew) / Number(anchorOld) : null);
    const runKey = String(r.run_id || `row-${i}`);
    const detailId = `runDetail${i}`;
    const isOpen = openRunDetails.has(runKey);
    const rows = fmt(r.new_rows_used ?? r.pending_rows, 0);
    return `<tr class="history-summary-row">`
      + `<td><div class="history-time"><b>${esc(r.time || '')}</b><small>${esc(r.run_id || '')}</small></div></td>`
      + `<td>${runStatusBadge(r.status)}</td>`
      + `<td>${r.version == null ? '—' : `v${esc(r.version)}`}</td>`
      + `<td><span class="gate-pill">${esc(gate)}</span></td>`
      + `<td>${formatMseChange(blendedOld, blendedNew)}</td>`
      + `<td>${anchorReg == null ? '—' : `<span class="anchor-reg ${Number(anchorReg) <= 1.08 ? 'ok' : 'bad'}">${fmtFixed(anchorReg, 4)}×</span>`}</td>`
      + `<td><div class="metric-stack compact"><b>${rows}</b><small>new rows</small></div></td>`
      + `<td class="history-reason">${esc(compactReason(r))}</td>`
      + `<td><button class="mini detail-toggle" type="button" data-run-key="${esc(runKey)}" data-detail="${detailId}">${isOpen ? 'Hide' : 'Details'}</button></td>`
      + `</tr>`
      + `<tr id="${detailId}" class="history-detail-row ${isOpen ? '' : 'is-hidden'}"><td colspan="9">`
      + `<div class="history-detail-grid">`
      + `<article><span>Anchor MSE</span><b>${formatMseChange(anchorOld, anchorNew)}</b><small>Fixed old-distribution validation</small></article>`
      + `<article><span>Recent MSE</span><b>${fmtRecentMse(recentOld, gate)} → ${fmtRecentMse(recentNew, gate)}</b><small>${String(gate).toLowerCase() === 'anchor_only' ? 'Not used in this run' : 'Recent drift-aware holdout'}</small></article>`
      + `<article><span>Blended MSE</span><b>${formatMseChange(blendedOld, blendedNew)}</b><small>Promotion decision score</small></article>`
      + `<article><span>Run Data</span><b>${fmt(r.train_rows, 0)} train · ${rows} new</b><small>${esc(r.reason || '')}</small></article>`
      + `</div></td></tr>`;
  }).join('');
  body.innerHTML = runningRow + (rowsHtml || (!running ? '<tr><td colspan="9" class="empty">No retraining history yet.</td></tr>' : ''));
  body.querySelectorAll('.detail-toggle').forEach(btn => btn.addEventListener('click', () => {
    const row = document.getElementById(btn.dataset.detail);
    if (!row) return;
    row.classList.toggle('is-hidden');
    if (row.classList.contains('is-hidden')) {
      openRunDetails.delete(btn.dataset.runKey);
      btn.textContent = 'Details';
    } else {
      openRunDetails.add(btn.dataset.runKey);
      btn.textContent = 'Hide';
    }
  }));
}


async function startRetrain(force) {
  const f = lastState?.freshness || {};
  if (force && !f.ready) {
    const ok = confirm(`Only ${f.new_rows || 0}/${f.threshold || 30} clean rows are ready. Force retrain is an operator override. The current queued rows will be locked as one retraining batch. Rows uploaded after the run starts will remain queued for the next retrain. Continue?`);
    if (!ok) return;
  }
  const btn = force ? $('forceRetrainBtn') : $('retrainBtn');
  if (btn) setButtonBusy(btn, true, 'Queueing…');
  try {
    const r = await fetch('/api/admin/retrain', { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify({ force }) });
    const d = await r.json();
    if (!d.ok) {
      alert(d.error || 'Could not start retraining.');
      return;
    }
    poll();
  } finally {
    if (btn) setButtonBusy(btn, false);
    loadState();
  }
}
$('refreshBtn')?.addEventListener('click', loadState);
$('retrainBtn')?.addEventListener('click', () => startRetrain(false));
$('forceRetrainBtn')?.addEventListener('click', () => startRetrain(true));

$('uploadBtn')?.addEventListener('click', async () => {
  const file = selectedCsvFile || $('csvFile')?.files?.[0];
  if (!file) { setUploadMessage('Choose or drop a CSV first.', 'error'); return; }
  if (!isCsvFile(file)) { setUploadMessage('Please upload a .csv file.', 'error'); return; }
  const fd = new FormData(); fd.append('file', file);
  setUploadBusy(true);
  setUploadMessage('Uploading CSV and running Gate 1 validation…', 'info', true);
  try {
    const r = await fetch('/api/admin/upload', { method: 'POST', headers: authHeaders(), body: fd });
    const d = await r.json();
    if (!d.ok) { setUploadMessage(d.error || 'Upload failed.', 'error'); return; }
    const uploadedName = file.name || 'CSV file';
    const autoRetrainStarted = Boolean(d.auto_retrain && d.auto_retrain.triggered);
    const autoNote = autoRetrainStarted ? ' Auto-retraining has been queued, so clean rows may move from Queued Clean Rows into the training run.' : '';
    resetCsvPicker();
    setUploadBusy(false);
    setUploadMessage(`Last upload: ${uploadedName} · Accepted ${d.accepted} rows · ${d.rejected} sent to review · queued clean rows now ${d.pending_total}.${autoNote}`, 'success');
    showToast(autoRetrainStarted ? `CSV uploaded · ${d.accepted} accepted · retraining queued.` : `CSV uploaded · ${d.accepted} accepted, ${d.rejected} to review.`, 'success');
    const pendingBody = document.querySelector('#pendingTable tbody');
    if (pendingBody && Number(d.accepted || 0) > 0) {
      pendingBody.innerHTML = `<tr><td colspan="6" class="empty">${autoRetrainStarted ? 'Upload accepted. A locked retraining batch is running; any new rows stay queued for the next run.' : 'Upload accepted. Refreshing queued clean rows preview…'}</td></tr>`;
    }
    loadState();
  } catch (_) {
    setUploadMessage('Upload failed. Please check the CSV file and try again.', 'error');
  } finally {
    setUploadBusy(false);
  }
});

function poll() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    try {
      const r = await fetch('/api/admin/state', { headers: authHeaders() });
      const d = await r.json();
      renderState(d);
      if (d.status && String(d.status.state || '').toLowerCase() !== 'running') {
        clearInterval(pollTimer);
        pollTimer = null;
        await loadState();
        loadGA(true);
        loadMembership(currentMembershipView, true);
      }
    } catch (_) {}
  }, 2500);
}

async function loadGA(force = false) {
  if (gaLoaded && !force) return;
  gaLoaded = true;
  try {
    const response = await fetch('/api/ga', { headers: authHeaders() });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || 'Could not load GA data');
    const s = data.stats || {};
    if ($('gaMetrics')) {
      const showFit = (v) => fmtFixed(v, 2);
      $('gaMetrics').innerHTML = [
        metricCard('GA Objective', `${showFit(s.initial_fitness)} → ${showFit(s.final_fitness)}`, 'training fitness; lower is better'),
        metricCard('Fitness Drop', `${esc(s.improvement_pct)}%`, 'internal GA improvement'),
        metricCard('Training Records', fmt(s.rows, 0), `${fmt(s.approved, 0)} approved · ${fmt(s.rejected, 0)} rejected`),
        metricCard('Promotion Gate', 'Blended MSE', 'production decision shown in Versions'),
      ].join('');
    }
    const fig = data.figure || { data: [], layout: {} };
    fig.layout = { ...(fig.layout || {}), annotations: [], autosize: true, height: 410, margin: { l: 54, r: 24, t: 44, b: 50 }, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: { family: 'Inter, sans-serif', color: '#596579' } };
    await Plotly.newPlot('gaChart', fig.data || [], fig.layout, plotConfig);
    if ($('breakpointTable')) $('breakpointTable').innerHTML = `<table><thead><tr><th>Variable</th><th>Low / Start</th><th>Mid / Peak</th><th>High / End</th><th>Unit</th></tr></thead><tbody>${(data.breakpoints || []).map(row => { const unit = String(row.variable || '').includes('Income') ? 'RM' : String(row.variable || '').includes('Ratio') ? '0–1' : 'years'; return `<tr><td>${esc(row.variable)}</td><td>${fmt(row.a, 4)}</td><td>${fmt(row.b, 4)}</td><td>${fmt(row.c, 4)}</td><td>${unit}</td></tr>`; }).join('')}</tbody></table>`;
    resizePlots();
  } catch (e) { gaLoaded = false; if ($('gaChart')) $('gaChart').innerHTML = `<div class="error-box">Could not load GA data. ${esc(e.message)}</div>`; }
}
$('reloadGaBtn')?.addEventListener('click', () => loadGA(true));
$('rollbackV0Btn')?.addEventListener('click', async () => {
  if (!confirm('Rollback to v0 baseline?')) return;
  await fetch('/api/admin/rollback', { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify({ version: 0 }) });
  loadState(); loadMembership(currentMembershipView, true);
});
$('resetStateBtn')?.addEventListener('click', async () => {
  const ok = confirm('Reset demo learning state? This clears pending/review rows, accumulated uploaded data, promoted versions except v0, retraining history, and returns the active model to v0.');
  if (!ok) return;
  const typed = prompt('Type RESET to confirm.');
  if (typed !== 'RESET') return;
  const r = await fetch('/api/admin/reset-state', { method: 'POST', headers: authHeaders({ 'Content-Type': 'application/json' }), body: JSON.stringify({ confirm: 'RESET' }) });
  const d = await r.json();
  if (!d.ok) { alert(d.error || 'Reset failed.'); return; }
  gaLoaded = false; mfLoaded = false;
  await loadState(); loadGA(true); loadMembership(currentMembershipView, true);
});

async function loadMembership(view = 'all', force = false) {
  if (mfLoaded && !force && currentMembershipView === view) return;
  mfLoaded = true; currentMembershipView = view;
  document.querySelectorAll('.mf-filter').forEach(btn => btn.classList.toggle('active', btn.dataset.view === view));
  const titles = { all: ['All Membership Functions', 'Complete fuzzy system including inputs and output risk score.'], ga: ['GA-Tuned Membership Functions', 'Annual Income, Loan-to-Income Ratio, and Employment Experience.'], fixed: ['Fixed Membership Functions', 'CTOS Credit Score, Previous Loan Default, and Output Risk Score.'] };
  const copy = titles[view] || titles.all;
  if ($('membershipTitle')) $('membershipTitle').textContent = copy[0];
  if ($('membershipSubtitle')) $('membershipSubtitle').textContent = copy[1];
  if ($('membershipChart')) $('membershipChart').innerHTML = '<div class="loading-text">Loading membership functions…</div>';
  try {
    const response = await fetch(`/api/membership?view=${encodeURIComponent(view)}`, { headers: authHeaders() });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || 'Could not load membership data');
    const chart = $('membershipChart');
    const charts = Array.isArray(data.charts) ? data.charts : [];
    if (chart && charts.length) {
      chart.removeAttribute('style');
      chart.className = 'membership-card-grid';
      chart.innerHTML = charts.map((card, idx) => `
        <article class="mf-chart-card">
          <div class="mf-card-head"><div><span class="mf-tag kind-${esc(card.kind)}">${esc(card.tag)}</span><h3>${esc(card.title)}</h3><p>${esc(card.subtitle || '')}</p></div></div>
          <div id="mfCardChart${idx}" class="mf-card-chart"></div>
        </article>
      `).join('');
      await new Promise(resolve => requestAnimationFrame(resolve));
      for (let i = 0; i < charts.length; i++) {
        const el = document.getElementById(`mfCardChart${i}`);
        const fig = charts[i].figure || { data: [], layout: {} };
        const width = Math.max(320, Math.floor(el?.getBoundingClientRect().width || 560));
        fig.layout = { ...(fig.layout || {}), autosize: false, width, height: 285, margin: { l: 44, r: 16, t: 10, b: 38 }, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: { family: 'Inter, sans-serif', color: '#596579' } };
        await Plotly.newPlot(`mfCardChart${i}`, fig.data || [], fig.layout, plotConfig);
      }
      setTimeout(resizePlots, 120);
    } else {
      const fig = data.figure || { data: [], layout: {} };
      const figHeight = (fig.layout && fig.layout.height) ? Number(fig.layout.height) : 700;
      if (chart) {
        chart.innerHTML = '';
        chart.style.height = `${figHeight}px`;
        chart.style.minHeight = `${figHeight}px`;
      }
      fig.layout = { ...(fig.layout || {}), autosize: true, height: figHeight, margin: { l: 48, r: 22, t: 42, b: 42 }, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: { family: 'Inter, sans-serif', color: '#596579' } };
      await Plotly.newPlot('membershipChart', fig.data || [], fig.layout, plotConfig);
      resizePlots();
    }
  } catch (e) { mfLoaded = false; if ($('membershipChart')) $('membershipChart').innerHTML = `<div class="error-box">Could not load membership functions. ${esc(e.message)}</div>`; }
}
document.querySelectorAll('.mf-filter').forEach(btn => btn.addEventListener('click', () => loadMembership(btn.dataset.view || 'all', true)));

loadState();

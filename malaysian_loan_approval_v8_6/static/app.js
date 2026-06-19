const plotConfig = { displayModeBar: false, responsive: true };
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function escapeHtml(value) {
  return String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}
// Safe markdown-lite: escape first, THEN apply our own tags, so AI text can
// never inject HTML. Supports **bold**, *italic*, "- " bullets and paragraphs.
function renderSummary(text) {
  const esc = escapeHtml(String(text ?? '')).trim();
  if (!esc) return '';
  const inline = (s) => s
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>');
  let html = '', inList = false;
  for (const raw of esc.split(/\r?\n/)) {
    const line = raw.trim();
    if (/^[-•]\s+/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${inline(line.replace(/^[-•]\s+/, ''))}</li>`;
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      if (line) html += `<p>${inline(line)}</p>`;
    }
  }
  if (inList) html += '</ul>';
  return html;
}
function setProcess(stepIndex, state='running') {
  const steps = Array.from(document.querySelectorAll('.process-step'));
  const status = document.getElementById('processStatus');
  const fill = document.getElementById('processFill');
  steps.forEach((step, idx) => {
    const circle = step.querySelector('span');
    step.classList.remove('active','done');
    circle.textContent = String(idx + 1);
    if (idx < stepIndex || state === 'done') { step.classList.add('done'); circle.textContent = '✓'; }
    if (idx === stepIndex && state === 'running') step.classList.add('active');
  });
  if (status) { status.textContent = state === 'done' ? 'Completed' : state === 'running' ? 'Processing' : 'Ready'; status.classList.toggle('done', state === 'done'); }
  // Fill spans the step circle centres only (left:10% .. up to 80% wide -> 90%),
  // so the line stays inside the workflow container regardless of step count.
  if (fill) {
    const lastStep = Math.max(1, steps.length - 1);
    const reached = state === 'done' ? lastStep : Math.max(0, Math.min(lastStep, stepIndex));
    fill.style.width = `${(reached / lastStep) * 80}%`;
  }
}
function toneClass(decision) { return decision === 'APPROVE' ? 'approve' : decision === 'REVIEW' ? 'review' : 'reject'; }
function toneText(decision) { return `${toneClass(decision)}-text`; }
function riskLevel(score) { return score < 40 ? 'Low' : score < 70 ? 'Medium' : 'High'; }
function recommendation(decision) { return decision === 'APPROVE' ? 'Approve loan' : decision === 'REVIEW' ? 'Manual review required' : 'Reject / restructure'; }

function calculateRatio() {
  const form = document.getElementById('loanForm');
  const income = Number(form.elements.income.value || 0);
  const loanAmount = Number(form.elements.loan_amount.value || 0);
  if (!income || income <= 0) return NaN;
  return loanAmount / income;
}
function updateRatioPreview() {
  const ratio = calculateRatio();
  const value = document.getElementById('ratioValue');
  const warning = document.getElementById('ratioWarning');
  const submitBtn = document.querySelector('#loanForm button[type="submit"]');
  let invalid = false;
  if (!Number.isFinite(ratio)) { value.textContent = '--'; warning.textContent = 'Annual income must be greater than 0.'; invalid = true; }
  else { value.textContent = ratio.toFixed(2); if (ratio < 0 || ratio > 1) { warning.textContent = 'Please reduce the loan amount. Debt-to-income ratio should be between 0.00 and 1.00.'; invalid = true; } else warning.textContent = ''; }
  warning.classList.toggle('visible', invalid);
  if (submitBtn) submitBtn.disabled = invalid;
}
function payloadFromForm(form) {
  const income = Number(form.elements.income.value);
  const loanAmount = Number(form.elements.loan_amount.value);
  const ratio = income > 0 ? loanAmount / income : NaN;
  return { income, loan_amount: loanAmount, credit: Number(form.elements.credit.value), ratio: Number.isFinite(ratio) ? Number(ratio.toFixed(4)) : ratio, emp: Number(form.elements.emp.value), default: String(form.elements.default.value) };
}
function setForm(values) {
  const form = document.getElementById('loanForm');
  Object.entries(values).forEach(([key, val]) => { if (form.elements[key]) form.elements[key].value = val; });
  updateRatioPreview();
}
// Fallback profiles (used until the engine-aware samples load). These are
// replaced at runtime by /api/samples so each button reliably lands in its
// decision band under whatever model is currently served.
let presets = {
  approve: { income: 65000, credit: 720, loan_amount: 11700, emp: 4, default: 'No' },
  review: { income: 50000, credit: 640, loan_amount: 20000, emp: 2, default: 'No' },
  reject: { income: 32000, credit: 520, loan_amount: 17600, emp: 1, default: 'Yes' },
};
document.querySelectorAll('.preset').forEach(btn => btn.addEventListener('click', () => setForm(presets[btn.dataset.preset])));

// Pull samples calibrated to the live fuzzy engine (tracks GA retraining).
fetch('/api/samples').then(r => r.json()).then(d => {
  if (d && d.ok && d.samples) {
    ['approve', 'review', 'reject'].forEach(k => { if (d.samples[k]) presets[k] = d.samples[k]; });
  }
}).catch(() => {});

function renderGauge(score, decision) {
  const color = decision === 'APPROVE' ? '#0f8a5f' : decision === 'REVIEW' ? '#b7791f' : '#c2413d';
  const data = [{
    type: 'indicator', mode: 'gauge+number', value: score,
    number: { font: { size: 36, color }, suffix: ' /100' },
    gauge: {
      shape: 'angular', axis: { range: [0, 100], tickvals: [0,40,70,100], tickfont: { size: 10, color: '#64748b' } },
      bar: { color, thickness: .22 }, bgcolor: '#f8fafc', borderwidth: 0,
      steps: [ { range:[0,40], color:'rgba(15,138,95,.22)' }, { range:[40,70], color:'rgba(183,121,31,.24)' }, { range:[70,100], color:'rgba(194,65,61,.20)' } ],
      threshold: { line: { color, width: 3 }, thickness: .8, value: score }
    }
  }];
  const layout = { height: 220, margin: { l: 10, r: 10, t: 10, b: 0 }, paper_bgcolor: '#fff', font: { family: 'Inter, sans-serif' } };
  Plotly.react('riskGauge', data, layout, plotConfig);
}
function factorRow(label, status, pct, tone='neutral') {
  const width = Math.max(0, Math.min(100, Number(pct) || 0));
  return `<div class="factor-row ${tone}"><div class="factor-top"><span>${escapeHtml(label)}</span><b>${escapeHtml(status)}</b></div><div class="bar-track"><span style="width:${width}%"></span></div></div>`;
}
function reasonRows(reasons) {
  return (reasons || []).map(r => `<div class="reason-item"><span class="reason-dot ${escapeHtml(r.tone || 'neutral')}"></span><span>${escapeHtml(r.text)}</span></div>`).join('') || '<div class="reason-item"><span>Assessment completed.</span></div>';
}
function whyRows(factors) {
  return (factors || []).map(f => `<div class="why-item"><span class="why-dot ${escapeHtml(f.tone || 'neutral')}"></span><span>${escapeHtml(f.text)}</span></div>`).join('') || '<div class="why-item"><span>Balanced profile across all five factors.</span></div>';
}
function buildDecisionDrivers(whyFactors, reasons) {
  const seen = new Set();
  const positive = [];
  const risk = [];
  const push = (item) => {
    const text = String(item?.text || '').trim();
    if (!text) return;
    const key = text.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    const tone = item?.tone || 'neutral';
    const row = `<div class="driver-item ${escapeHtml(tone)}"><span></span><p>${escapeHtml(text)}</p></div>`;
    if (tone === 'positive') positive.push(row);
    else risk.push(row);
  };
  (whyFactors || []).forEach(push);
  (reasons || []).forEach(push);
  const positiveHtml = positive.length ? positive.join('') : '<div class="driver-empty">No strong positive factor detected.</div>';
  const riskHtml = risk.length ? risk.join('') : '<div class="driver-empty">No major risk factor detected.</div>';
  return `
    <div class="driver-card positive"><h3>Positive Factors</h3>${positiveHtml}</div>
    <div class="driver-card risk"><h3>Risk / Review Factors</h3>${riskHtml}</div>`;
}
// Holds the most recent assessment payload so "Export Assessment PDF" can send
// exactly what the user sees to the server-side PDF generator.
let lastAssessment = null;
function renderWhatIf(items, decision) {
  if (items && items.length) {
    return items.map(item => `<div class="whatif-item"><div><b>${escapeHtml(item.label)}</b><p>${escapeHtml(item.advice)}</p></div><div class="whatif-score">${escapeHtml(item.delta_text || `Risk can be reduced by ${item.delta} points`)}<small>${escapeHtml(item.new_score)} / 100</small></div></div>`).join('');
  }
  if (decision === 'APPROVE') return '<div class="whatif-item"><div><b>Maintain current profile</b><p>Applicant is already within the low-risk band.</p></div><div class="whatif-score">OK<small>Stable</small></div></div>';
  return '<div class="whatif-item"><div><b>Major restructuring needed</b><p>This profile needs major restructuring — consider reducing the loan amount significantly.</p></div><div class="whatif-score">Action<small>Required</small></div></div>';
}
function renderResult(data) {
  const { applicant, ctos, result, reasons, what_if } = data;
  lastAssessment = data;
  const summaryHtml = renderSummary(data.agent_summary);
  const aiProvider = data.ai_provider || null;
  const aiError = data.ai_error || null;
  const disclaimer = data.disclaimer || 'This system supports loan assessment decisions but does not replace final human officer review.';
  const d = result.membership_degrees || {};
  const decision = result.decision;
  const tone = toneClass(decision);
  const score = Number(result.risk_score);
  const loanAmount = Number(applicant.loan_amount || applicant.income * applicant.ratio || 0);
  const factorHtml = [
    factorRow('Income Strength', d.income_high >= .5 ? 'High' : d.income_medium >= .5 ? 'Moderate' : 'Low', Math.max(d.income_high||0,d.income_medium||0,d.income_low||0)*100, d.income_high >= .5 ? 'positive' : 'neutral'),
    factorRow('Debt Pressure', d.ratio_high >= .5 ? 'High' : d.ratio_medium >= .5 ? 'Moderate' : 'Low', Math.max(d.ratio_high||0,d.ratio_medium||0,d.ratio_low||0)*100, d.ratio_high >= .5 ? 'negative' : 'positive'),
    factorRow('Employment Stability', d.emp_experienced >= .5 ? 'Experienced' : d.emp_mid >= .5 ? 'Mid-level' : 'Junior', Math.max(d.emp_experienced||0,d.emp_mid||0,d.emp_junior||0)*100, d.emp_junior >= .5 ? 'neutral' : 'positive'),
    factorRow('Credit Risk', ctos.category || '—', Math.round((1 - Number(d.credit_risk || 0))*100), Number(d.credit_risk || 0) <= .25 ? 'positive' : Number(d.credit_risk || 0) >= .55 ? 'negative' : 'neutral'),
    factorRow('Default History', Number(d.defaulted) ? 'Previous default' : 'Clear', Number(d.defaulted) ? 86 : 24, Number(d.defaulted) ? 'negative' : 'positive'),
  ].join('');
  const aiMetaHtml = aiProvider
    ? `<span class="ai-provider-pill">AI Provider Used: ${escapeHtml(aiProvider)}</span>`
    : `<span class="ai-provider-pill off">AI Provider Used: Unavailable</span>`;
  const explanationHtml = aiError
    ? `<div class="section-title">AI Officer Explanation</div><div class="ai-unavailable">${escapeHtml(aiError)}</div>`
    : (summaryHtml ? `<div class="section-title">AI Officer Explanation</div><div class="decision-explanation">${summaryHtml}</div>` : '');
  document.getElementById('resultCard').innerHTML = `
    <div class="decision-banner ${tone}">
      <div><span class="decision-kicker">Final Decision</span><strong>${escapeHtml(decision)}</strong></div>
      <div class="decision-score"><span>Risk Score</span><b>${score.toFixed(1)} / 100</b><small>${riskLevel(score)} Risk · ${recommendation(decision)}</small></div>
    </div>
    <div class="result-top">
      <div class="gauge-wrap"><div id="riskGauge" class="riskGauge"></div><div class="risk-hint">Risk Score: Lower = Safer</div><div class="risk-legend"><span class="low">0–39 Approve</span><span class="mid">40–69 Review</span><span class="high">70–100 Reject</span></div></div>
      <div class="evidence-panel">
        <div class="section-title small">Assessment Evidence</div>
        <div class="info-grid">
          <div class="info-box"><div class="k">CTOS Evidence</div><div class="v">${Math.round(applicant.credit)} · ${escapeHtml(ctos.category)}</div></div>
          <div class="info-box"><div class="k">Requested Amount</div><div class="v">RM${loanAmount.toLocaleString()}</div></div>
          <div class="info-box"><div class="k">Debt-to-Income</div><div class="v">${Number(applicant.ratio).toFixed(2)}</div></div>
          <div class="info-box"><div class="k">Employment</div><div class="v">${Number(applicant.emp).toFixed(1)} years</div></div>
          <div class="info-box wide"><div class="k">Previous Default</div><div class="v">${escapeHtml(applicant.default || 'No')}</div></div>
        </div>
      </div>
    </div>
    <div class="section-title">Decision Drivers</div><div class="driver-grid">${buildDecisionDrivers(data.why_factors, reasons)}</div>
    <div class="section-title">Risk Factor Profile</div><div class="factor-list compact">${factorHtml}</div>
    <div class="section-title">What-If Improvement Options</div><div class="whatif-list ranked">${renderWhatIf(what_if, decision)}</div>
    <div class="ai-meta">${aiMetaHtml}</div>
    ${explanationHtml}
    <div class="disclaimer"><span>ⓘ</span><b>${escapeHtml(disclaimer)}</b></div>
    <div class="result-actions"><button class="export-btn" id="exportPdfBtn" type="button"><span>⤓</span>Export Assessment PDF</button></div>`;
  renderGauge(score, decision);
  const exportBtn = document.getElementById('exportPdfBtn');
  if (exportBtn) exportBtn.addEventListener('click', exportAssessmentPdf);
}

async function exportAssessmentPdf() {
  if (!lastAssessment) return;
  const btn = document.getElementById('exportPdfBtn');
  const original = btn ? btn.innerHTML : '';
  if (btn) { btn.disabled = true; btn.innerHTML = '<span>⤓</span>Preparing PDF…'; }
  try {
    const response = await fetch('/api/assessment/pdf', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(lastAssessment) });
    if (!response.ok) throw new Error('PDF export failed');
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'loan_assessment.pdf';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  } catch (err) {
    if (btn) { btn.innerHTML = '<span>⤓</span>Export failed — try again'; setTimeout(() => { btn.innerHTML = original; btn.disabled = false; }, 2200); return; }
  } finally {
    if (btn) { btn.innerHTML = original; btn.disabled = false; }
  }
}
function renderError(message, isValidation=false, code='') {
  const isAiRequired = code === 'AI_REQUIRED' || code === 'AI_UNAVAILABLE' || code === 'TOOL_CALL_REQUIRED';
  const title = isValidation ? 'Input Check' : isAiRequired ? 'AI Reasoning Key Required' : 'Analysis temporarily unavailable';
  const display = isValidation || isAiRequired ? message : 'Analysis temporarily unavailable. Please try again.';
  document.getElementById('resultCard').innerHTML = `<div class="error-box ${isAiRequired ? 'ai-key-error' : ''}"><b>${escapeHtml(title)}</b>
${escapeHtml(display)}</div>`;
}

document.getElementById('loanForm').addEventListener('submit', async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector('button[type="submit"]');
  const payload = payloadFromForm(form);
  updateRatioPreview();
  if (!Number.isFinite(payload.ratio) || payload.ratio < 0 || payload.ratio > 1) { renderError('Debt-to-Income Ratio must be between 0.00 and 1.00. Please reduce the requested loan amount or check the income value.', true); return; }
  button.disabled = true; button.textContent = 'Evaluating…';
  document.getElementById('resultCard').innerHTML = '<div class="loading-text">Running applicant risk assessment…</div>';
  try {
    setProcess(0,'running'); await sleep(140); setProcess(1,'running'); await sleep(140); setProcess(2,'running');
    const response = await fetch('/api/evaluate', { method:'POST', headers:{ 'Content-Type':'application/json' }, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok || !data.ok) { const err = new Error(data.error || 'Evaluation failed'); err.validation = data.code === 'VALIDATION'; err.code = data.code || ''; throw err; }
    setProcess(3,'running'); await sleep(120); renderResult(data); setProcess(4,'running'); await sleep(120); setProcess(4,'done');
  } catch (err) { renderError(err.message || String(err), Boolean(err.validation), err.code || ''); setProcess(0,'idle'); }
  finally { button.disabled = false; button.textContent = 'Evaluate Applicant'; }
});
['income','loan_amount'].forEach(name => {
  document.getElementById('loanForm').elements[name].addEventListener('input', updateRatioPreview);
  document.getElementById('loanForm').elements[name].addEventListener('blur', updateRatioPreview);
});
updateRatioPreview();
setProcess(0,'idle');

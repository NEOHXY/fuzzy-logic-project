// Shared sidebar collapse + public model-version badge
(function () {
  const KEY = 'ML_SIDEBAR_COLLAPSED';

  function updateButton(collapsed) {
    const btn = document.getElementById('sidebarToggle');
    if (!btn) return;
    const icon = btn.querySelector('.material-symbols-outlined');
    if (icon) icon.textContent = collapsed ? 'chevron_right' : 'chevron_left';
    btn.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
    btn.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
  }

  function applySidebar(collapsed) {
    document.body.classList.toggle('sidebar-collapsed', Boolean(collapsed));
    updateButton(Boolean(collapsed));
    try { localStorage.setItem(KEY, collapsed ? '1' : '0'); } catch (_) {}
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
      if (window.Plotly) {
        document.querySelectorAll('.js-plotly-plot').forEach(el => {
          try { Plotly.Plots.resize(el); } catch (_) {}
        });
      }
    }, 260);
  }

  function updateVersionLabels(label) {
    if (!label) return;
    document.querySelectorAll('[data-model-version-label]').forEach(el => {
      el.textContent = label;
    });
  }

  function loadModelVersion() {
    fetch('/api/model-status')
      .then(r => r.json())
      .then(d => { if (d && d.ok) updateVersionLabels(d.active_version_label || `v${d.active_version || 0}`); })
      .catch(() => {});
  }

  document.addEventListener('DOMContentLoaded', () => {
    let collapsed = false;
    try { collapsed = localStorage.getItem(KEY) === '1'; } catch (_) {}
    applySidebar(collapsed);
    const btn = document.getElementById('sidebarToggle');
    if (btn) {
      btn.addEventListener('click', () => applySidebar(!document.body.classList.contains('sidebar-collapsed')));
    }
    loadModelVersion();
  });
})();

(function () {
  var cols = window.EDA_COLS || [];
  var kindOf = {};
  cols.forEach(function (c) { kindOf[c.name] = c.kind; });
  var selX = document.getElementById('selX'), selY = document.getElementById('selY');
  var list = document.getElementById('pairList'), empty = document.getElementById('pairEmpty');
  var kindLbl = document.getElementById('pairKind');
  var pairs = [];

  function fill(sel, preferred) {
    ['numeric', 'categorical'].forEach(function (k) {
      var g = document.createElement('optgroup');
      g.label = k === 'numeric' ? 'Numerical' : 'Categorical';
      cols.filter(function (c) { return c.kind === k; }).forEach(function (c) {
        var o = document.createElement('option'); o.value = c.name; o.textContent = c.name; g.appendChild(o);
      });
      if (g.children.length) sel.appendChild(g);
    });
  }
  fill(selX); fill(selY);
  if (selY.options.length > 1) selY.selectedIndex = 1;

  function label(x, y) {
    var a = kindOf[x], b = kindOf[y];
    if (a === 'numeric' && b === 'numeric') return 'Num vs Num';
    if (a === 'categorical' && b === 'categorical') return 'Cat vs Cat';
    return 'Num vs Cat';
  }
  function refreshKind() { kindLbl.textContent = selX.value && selY.value ? '(' + label(selX.value, selY.value) + ')' : ''; }
  selX.addEventListener('change', refreshKind); selY.addEventListener('change', refreshKind); refreshKind();

  function render() {
    list.innerHTML = '';
    pairs.forEach(function (p, i) {
      var li = document.createElement('li');
      li.className = 'list-group-item d-flex align-items-center';
      var badge = document.createElement('span'); badge.className = 'badge text-bg-secondary me-2'; badge.textContent = label(p.x, p.y);
      var txt = document.createElement('span'); txt.className = 'flex-grow-1 text-break'; txt.textContent = p.x + '  ×  ' + p.y;
      var hx = document.createElement('input'); hx.type = 'hidden'; hx.name = 'pair_x'; hx.value = p.x;
      var hy = document.createElement('input'); hy.type = 'hidden'; hy.name = 'pair_y'; hy.value = p.y;
      var rm = document.createElement('button'); rm.type = 'button'; rm.className = 'btn btn-sm btn-outline-danger';
      rm.innerHTML = '<i class="bi bi-x-lg"></i>';
      rm.addEventListener('click', function () { pairs.splice(i, 1); render(); });
      li.appendChild(badge); li.appendChild(txt); li.appendChild(hx); li.appendChild(hy); li.appendChild(rm);
      list.appendChild(li);
    });
    empty.style.display = pairs.length ? 'none' : '';
  }
  function add(x, y) {
    if (!x || !y || x === y) { alert('Please choose two different columns.'); return; }
    if (pairs.some(function (p) { return p.x === x && p.y === y; })) return;
    if (pairs.length >= window.EDA_MAX_PAIRS) { alert('Maximum ' + window.EDA_MAX_PAIRS + ' pairs.'); return; }
    pairs.push({ x: x, y: y }); render();
  }
  document.getElementById('addPair').addEventListener('click', function () { add(selX.value, selY.value); });
  document.querySelectorAll('.suggest').forEach(function (b) {
    b.addEventListener('click', function () { add(b.dataset.x, b.dataset.y); });
  });
  (window.EDA_SAVED_PAIRS || []).forEach(function (p) { if (kindOf[p.x] && kindOf[p.y]) pairs.push(p); });
  render();

  // pair-plot column cap
  var pp = document.querySelectorAll('.pp'), cnt = document.getElementById('ppCount');
  function ppUpdate(e) {
    var n = document.querySelectorAll('.pp:checked').length;
    if (n > window.EDA_MAX_PP && e && e.target) { e.target.checked = false; n--; }
    if (cnt) cnt.textContent = n + ' selected' + (n < 2 ? ' (need at least 2 to draw a pair plot)' : '');
  }
  pp.forEach(function (c) { c.addEventListener('change', ppUpdate); });
  ppUpdate();
})();

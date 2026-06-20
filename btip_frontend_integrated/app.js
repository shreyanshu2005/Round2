(() => {
  // ─── CONFIG ────────────────────────────────────────────────────────────────
  const API_BASE = 'http://localhost:8000/api/v1';

  // ─── UTILS ─────────────────────────────────────────────────────────────────
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  // ─── API HELPERS ────────────────────────────────────────────────────────────
  let _jwt = null;

  async function getToken() {
    if (_jwt) return _jwt;
    try {
      const res = await fetch(`${API_BASE.replace('/api/v1', '')}/auth/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'username=commander1&password=commander123',
        credentials: 'omit'
      });
      if (res.ok) {
        const data = await res.json();
        _jwt = data.access_token;
        return _jwt;
      }
    } catch (_) { /* auth endpoint optional — proceed unauthenticated */ }
    return null;
  }

  async function apiFetch(path, options = {}) {
    const token = await getToken();
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const res = await fetch(`${API_BASE}${path}`, { ...options, headers, credentials: 'omit' });
    if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
    return res.json();
  }

  // ─── STATE ──────────────────────────────────────────────────────────────────
  const state = {
    cityRisk: 72,
    hotspots: [],
    recommendations: [],
    zones: ['silk', 'bellandur', 'hebbal', 'tin'],
    zoneIds: {}  // populated from hotspots response
  };

  // ─── SMOOTH SCROLL ──────────────────────────────────────────────────────────
  $$('[data-jump]').forEach((button) => {
    button.addEventListener('click', () => {
      const target = $(button.dataset.jump);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  // ─── LIVE CLOCK ─────────────────────────────────────────────────────────────
  const signalTime = $('#signalTime');
  const updateClock = () => {
    signalTime.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
  updateClock();
  setInterval(updateClock, 30_000);

  // ─── HEADER SCROLL ──────────────────────────────────────────────────────────
  const topbar = $('#topbar');
  window.addEventListener('scroll', () => topbar.classList.toggle('scrolled', scrollY > 40), { passive: true });

  // ─── REVEAL ─────────────────────────────────────────────────────────────────
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  $$('.reveal').forEach((el, index) => {
    el.style.transitionDelay = `${Math.min(index % 4, 3) * 70}ms`;
    revealObserver.observe(el);
  });

  // ─── CURSOR ─────────────────────────────────────────────────────────────────
  const cursor = $('.cursor-orbit');
  if (matchMedia('(pointer:fine)').matches) {
    window.addEventListener('pointermove', (event) => {
      cursor.style.left = `${event.clientX}px`;
      cursor.style.top = `${event.clientY}px`;
    });
    $$('a, button, input, select').forEach((el) => {
      el.addEventListener('pointerenter', () => cursor.classList.add('is-hover'));
      el.addEventListener('pointerleave', () => cursor.classList.remove('is-hover'));
    });
  }

  // ─── MOTION TOGGLE ──────────────────────────────────────────────────────────
  const motionToggle = $('#motionToggle');
  motionToggle.addEventListener('click', () => {
    const off = document.body.classList.toggle('motion-off');
    motionToggle.setAttribute('aria-pressed', String(off));
    motionToggle.lastChild.textContent = off ? ' Motion off' : ' Motion on';
  });

  // ─── MAGNETIC BUTTONS ───────────────────────────────────────────────────────
  $$('.magnetic').forEach((button) => {
    button.addEventListener('pointermove', (event) => {
      const rect = button.getBoundingClientRect();
      const x = (event.clientX - rect.left - rect.width / 2) * 0.08;
      const y = (event.clientY - rect.top - rect.height / 2) * 0.08;
      button.style.transform = `translate(${x}px, ${y}px)`;
    });
    button.addEventListener('pointerleave', () => { button.style.transform = ''; });
  });

  // ─── HERO PARTICLES ─────────────────────────────────────────────────────────
  const heroCanvas = $('#heroCanvas');
  const heroCtx = heroCanvas.getContext('2d');
  let heroParticles = [];
  const resizeHero = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    heroCanvas.width = innerWidth * dpr;
    heroCanvas.height = $('#hero').offsetHeight * dpr;
    heroCanvas.style.width = `${innerWidth}px`;
    heroCanvas.style.height = `${$('#hero').offsetHeight}px`;
    heroCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    heroParticles = Array.from({ length: innerWidth < 700 ? 24 : 52 }, (_, i) => ({
      x: Math.random() * innerWidth,
      y: Math.random() * $('#hero').offsetHeight * .72,
      r: 1 + Math.random() * 4,
      speed: .15 + Math.random() * .35,
      phase: Math.random() * Math.PI * 2,
      type: i % 5 === 0 ? 'signal' : 'ink'
    }));
  };
  const drawHero = (time = 0) => {
    heroCtx.clearRect(0, 0, innerWidth, $('#hero').offsetHeight);
    heroParticles.forEach((p) => {
      p.x += p.speed;
      if (p.x > innerWidth + 20) p.x = -20;
      const y = p.y + Math.sin(time * .001 + p.phase) * 10;
      heroCtx.beginPath();
      heroCtx.fillStyle = p.type === 'signal' ? 'rgba(255,107,69,.72)' : 'rgba(16,36,44,.22)';
      heroCtx.arc(p.x, y, p.r, 0, Math.PI * 2);
      heroCtx.fill();
      if (p.type === 'signal') {
        heroCtx.beginPath();
        heroCtx.strokeStyle = 'rgba(255,107,69,.16)';
        heroCtx.arc(p.x, y, p.r + 9 + Math.sin(time * .003) * 3, 0, Math.PI * 2);
        heroCtx.stroke();
      }
    });
    requestAnimationFrame(drawHero);
  };
  resizeHero();
  drawHero();
  window.addEventListener('resize', resizeHero);

  // ─── TRAFFIC NETWORK CANVAS ─────────────────────────────────────────────────
  const networkCanvas = $('#networkCanvas');
  const networkCtx = networkCanvas.getContext('2d');
  let networkLayer = 'risk';
  let networkSize = { width: 0, height: 0, dpr: 1 };
  let roads = [], nodes = [], vehicles = [];

  const makeNetwork = () => {
    const rect = networkCanvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    networkCanvas.width = rect.width * dpr;
    networkCanvas.height = rect.height * dpr;
    networkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    networkSize = { width: rect.width, height: rect.height, dpr };
    roads = []; nodes = []; vehicles = [];
    const roadCount = rect.width < 600 ? 15 : 26;
    for (let i = 0; i < roadCount; i++) {
      const horizontal = i % 2 === 0;
      roads.push({
        horizontal,
        offset: horizontal ? (i / roadCount) * rect.height : (i / roadCount) * rect.width,
        amp: 18 + Math.random() * 58,
        phase: Math.random() * Math.PI * 2,
        width: i % 5 === 0 ? 3 : 1.2,
        speed: .0008 + Math.random() * .0015
      });
    }
    // Seed nodes from live hotspot data if available, else random
    const hotspotCount = Math.min(state.hotspots.length, 28);
    if (hotspotCount > 0) {
      const bbox = { minLat: 12.8, maxLat: 13.2, minLng: 77.4, maxLng: 77.8 };
      state.hotspots.slice(0, 28).forEach((h) => {
        const x = rect.width * ((h.centroid_lng - bbox.minLng) / (bbox.maxLng - bbox.minLng));
        const y = rect.height * (1 - (h.centroid_lat - bbox.minLat) / (bbox.maxLat - bbox.minLat));
        nodes.push({ x: clamp(x, 0.05 * rect.width, 0.95 * rect.width), y: clamp(y, 0.05 * rect.height, 0.95 * rect.height), risk: h.risk_score / 100, phase: Math.random() * Math.PI * 2 });
      });
      // pad with random if fewer than 12
      while (nodes.length < 12) {
        nodes.push({ x: rect.width * (.08 + Math.random() * .84), y: rect.height * (.1 + Math.random() * .8), risk: Math.random(), phase: Math.random() * Math.PI * 2 });
      }
    } else {
      for (let i = 0; i < 28; i++) {
        nodes.push({ x: rect.width * (.08 + Math.random() * .84), y: rect.height * (.1 + Math.random() * .8), risk: Math.random(), phase: Math.random() * Math.PI * 2 });
      }
    }
    for (let i = 0; i < 52; i++) {
      vehicles.push({ road: Math.floor(Math.random() * roads.length), t: Math.random(), speed: .001 + Math.random() * .0025, size: 1.5 + Math.random() * 2.5 });
    }
  };

  const roadPoint = (road, t, time) => {
    const { width, height } = networkSize;
    if (road.horizontal) {
      return { x: t * width, y: road.offset + Math.sin(t * 8 + road.phase + time * road.speed) * road.amp };
    }
    return { x: road.offset + Math.sin(t * 8 + road.phase + time * road.speed) * road.amp, y: t * height };
  };

  const drawNetwork = (time = 0) => {
    const { width, height } = networkSize;
    if (!width || !height) return requestAnimationFrame(drawNetwork);
    networkCtx.clearRect(0, 0, width, height);
    const bg = networkCtx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, networkLayer === 'flow' ? '#c6f1ff' : '#dff5ff');
    bg.addColorStop(1, networkLayer === 'violations' ? '#fff1bd' : '#bde9ff');
    networkCtx.fillStyle = bg;
    networkCtx.fillRect(0, 0, width, height);
    networkCtx.save();
    networkCtx.globalAlpha = .18;
    networkCtx.strokeStyle = '#10242c';
    for (let r = 80; r < 500; r += 44) {
      networkCtx.beginPath();
      networkCtx.ellipse(width * .68, height * .32, r * 1.2, r * .72, -.25, 0, Math.PI * 2);
      networkCtx.stroke();
    }
    networkCtx.restore();
    roads.forEach((road) => {
      networkCtx.beginPath();
      for (let step = 0; step <= 80; step++) {
        const p = roadPoint(road, step / 80, time);
        if (step === 0) networkCtx.moveTo(p.x, p.y);
        else networkCtx.lineTo(p.x, p.y);
      }
      networkCtx.strokeStyle = road.width > 2 ? 'rgba(16,36,44,.5)' : 'rgba(16,36,44,.22)';
      networkCtx.lineWidth = road.width;
      networkCtx.stroke();
    });
    nodes.forEach((node) => {
      const radius = 4 + node.risk * 16 + Math.sin(time * .002 + node.phase) * 2;
      let color = '#54d566';
      if (node.risk > .72) color = '#ff6b45';
      else if (node.risk > .45) color = '#ffc928';
      if (networkLayer === 'violations') color = node.risk > .55 ? '#d92816' : '#2f73ff';
      if (networkLayer === 'flow') color = '#2f73ff';
      networkCtx.beginPath();
      networkCtx.fillStyle = `${color}33`;
      networkCtx.arc(node.x, node.y, radius * 2.7, 0, Math.PI * 2);
      networkCtx.fill();
      networkCtx.beginPath();
      networkCtx.fillStyle = color;
      networkCtx.arc(node.x, node.y, radius * .55, 0, Math.PI * 2);
      networkCtx.fill();
    });
    vehicles.forEach((vehicle) => {
      const road = roads[vehicle.road];
      vehicle.t = (vehicle.t + vehicle.speed) % 1;
      const p = roadPoint(road, vehicle.t, time);
      networkCtx.beginPath();
      networkCtx.fillStyle = networkLayer === 'flow' ? '#2f73ff' : '#10242c';
      networkCtx.arc(p.x, p.y, vehicle.size, 0, Math.PI * 2);
      networkCtx.fill();
    });
    requestAnimationFrame(drawNetwork);
  };
  setTimeout(makeNetwork, 50);
  drawNetwork();
  window.addEventListener('resize', makeNetwork);

  $$('.layer-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      $$('.layer-chip').forEach((item) => item.classList.remove('is-active'));
      chip.classList.add('is-active');
      networkLayer = chip.dataset.layer;
    });
  });

  $('#mapFocusCard').addEventListener('click', () => {
    nodes.forEach((node, index) => { node.risk = index % 4 === 0 ? .95 : Math.random() * .7; });
  });

  // ─── TILT CARDS ─────────────────────────────────────────────────────────────
  $$('.tilt-card').forEach((card) => {
    card.addEventListener('pointermove', (event) => {
      const rect = card.getBoundingClientRect();
      const rx = ((event.clientY - rect.top) / rect.height - .5) * -7;
      const ry = ((event.clientX - rect.left) / rect.width - .5) * 7;
      card.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateY(-4px)`;
    });
    card.addEventListener('pointerleave', () => { card.style.transform = ''; });
  });

  // ─── NUMBER ANIMATION ───────────────────────────────────────────────────────
  const animateNumber = (element, to, decimals = 0, suffix = '') => {
    const from = parseFloat(element.textContent) || 0;
    const start = performance.now();
    const duration = 850;
    const tick = (now) => {
      const p = clamp((now - start) / duration, 0, 1);
      const eased = 1 - Math.pow(1 - p, 4);
      element.textContent = (from + (to - from) * eased).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  // ─── SIMULATION ─────────────────────────────────────────────────────────────
  const officerRange = $('#officerRange');
  const officerValue = $('#officerValue');
  const zoneSelect = $('#zoneSelect');
  const runSimulation = $('#runSimulation');
  let selectedShift = 'Evening';

  $$('.shift-control button').forEach((button) => {
    button.addEventListener('click', () => {
      $$('.shift-control button').forEach((item) => item.classList.remove('is-active'));
      button.classList.add('is-active');
      selectedShift = button.dataset.shift;
    });
  });
  officerRange.addEventListener('input', () => { officerValue.textContent = officerRange.value; });

  runSimulation.addEventListener('click', async () => {
    runSimulation.disabled = true;
    runSimulation.querySelector('span').textContent = 'SIMULATING…';

    const officers = Number(officerRange.value);
    const zoneKey = zoneSelect.value;
    // Map dropdown value to zone_id from hotspots, fall back to index
    const zoneNameMap = { silk: 'Silk Board', bellandur: 'Bellandur', hebbal: 'Hebbal', tin: 'Tin Factory' };
    const matchedHotspot = state.hotspots.find(h =>
      h.zone_name && h.zone_name.toLowerCase().includes(zoneNameMap[zoneKey].toLowerCase())
    ) || state.hotspots[0];
    const zoneId = matchedHotspot ? (matchedHotspot.cluster_id ?? matchedHotspot.zone_id ?? 1) : 1;

    try {
      const payload = {
        zone_allocations: [{ zone_id: zoneId, n_officers: officers }],
        shift: selectedShift,
        date: new Date().toISOString().split('T')[0]
      };
      const data = await apiFetch('/simulation', { method: 'POST', body: JSON.stringify(payload) });

      const reduction = data.reduction_pct ?? data.total_reduction_pct ?? 18.6;
      const relief = data.congestion_improvement_pct ?? reduction * 0.66;
      const junctions = data.affected_junction_count ?? 14;
      const afterRisk = Math.round(86 - reduction * 1.7);
      const p10 = data.confidence_band?.p10 ?? reduction * 0.76;
      const p50 = data.confidence_band?.p50 ?? reduction;
      const p90 = data.confidence_band?.p90 ?? reduction * 1.24;

      animateNumber($('#reductionValue'), reduction, 1);
      animateNumber($('#reliefValue'), relief, 1);
      animateNumber($('#junctionValue'), junctions, 0);
      animateNumber($('#afterRisk'), afterRisk, 0);
      animateNumber($('#p10Value'), p10, 1, '%');
      animateNumber($('#p50Value'), p50, 1, '%');
      animateNumber($('#p90Value'), p90, 1, '%');
    } catch (err) {
      console.warn('Simulation API failed, falling back to model estimate:', err);
      // Graceful fallback: replicate original formula
      const zoneFactor = { silk: 1, bellandur: .92, hebbal: .8, tin: .74 }[zoneKey];
      const shiftFactor = { Morning: .82, Afternoon: .68, Evening: 1, Night: .55 }[selectedShift];
      const reduction = clamp((7 + Math.log(officers + 1) * 3.5) * zoneFactor * shiftFactor, 4, 32);
      animateNumber($('#reductionValue'), reduction, 1);
      animateNumber($('#reliefValue'), reduction * .66, 1);
      animateNumber($('#junctionValue'), Math.round(5 + officers * .45 * zoneFactor), 0);
      animateNumber($('#afterRisk'), Math.round(86 - reduction * 1.7), 0);
      animateNumber($('#p10Value'), reduction * .76, 1, '%');
      animateNumber($('#p50Value'), reduction, 1, '%');
      animateNumber($('#p90Value'), reduction * 1.24, 1, '%');
    }

    $('#afterMap').animate([
      { filter: 'saturate(.7) brightness(.95)' },
      { filter: 'saturate(1.3) brightness(1.06)' },
      { filter: 'none' }
    ], { duration: 900, easing: 'ease-out' });
    runSimulation.disabled = false;
    runSimulation.querySelector('span').textContent = 'RUN SCENARIO';
  });

  // ─── BEFORE/AFTER DIVIDER ───────────────────────────────────────────────────
  const simVisual = $('.sim-visual');
  const afterMap = $('#afterMap');
  const compareHandle = $('#compareHandle');
  let comparing = false;
  const setCompare = (clientX) => {
    const rect = simVisual.getBoundingClientRect();
    const percent = clamp(((clientX - rect.left) / rect.width) * 100, 10, 90);
    afterMap.style.clipPath = `inset(0 0 0 ${percent}%)`;
    compareHandle.style.left = `${percent}%`;
  };
  compareHandle.addEventListener('pointerdown', (event) => {
    comparing = true;
    compareHandle.setPointerCapture(event.pointerId);
  });
  compareHandle.addEventListener('pointermove', (event) => { if (comparing) setCompare(event.clientX); });
  compareHandle.addEventListener('pointerup', () => { comparing = false; });
  simVisual.addEventListener('pointerdown', (event) => {
    if (event.target === compareHandle || compareHandle.contains(event.target)) return;
    setCompare(event.clientX);
  });

  // ─── RECOMMENDATION ACCORDION ───────────────────────────────────────────────
  $$('.rec-expand').forEach((button) => {
    button.addEventListener('click', () => {
      const card = button.closest('.recommendation-card');
      const open = card.classList.toggle('is-open');
      button.setAttribute('aria-expanded', String(open));
    });
  });

  // ─── ORBIT NODE CLICK ───────────────────────────────────────────────────────
  $$('.orbit-node').forEach((node) => {
    node.addEventListener('click', () => {
      $$('.orbit-node').forEach((n) => n.style.background = '');
      node.style.background = '#b8ff3d';
      node.animate([
        { transform: 'scale(1)' },
        { transform: 'scale(1.12) rotate(-4deg)' },
        { transform: 'scale(1)' }
      ], { duration: 500, easing: 'cubic-bezier(.2,.85,.25,1)' });
    });
  });

  // ─── CLOSING CANVAS ─────────────────────────────────────────────────────────
  const closingCanvas = $('#closingCanvas');
  const closingCtx = closingCanvas.getContext('2d');
  let closingSize = {};
  const resizeClosing = () => {
    const rect = closingCanvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    closingCanvas.width = rect.width * dpr;
    closingCanvas.height = rect.height * dpr;
    closingCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    closingSize = rect;
  };
  const drawClosing = (time = 0) => {
    const { width = 0, height = 0 } = closingSize;
    closingCtx.clearRect(0, 0, width, height);
    closingCtx.strokeStyle = 'rgba(255,255,255,.18)';
    closingCtx.lineWidth = 1;
    const spacing = 54;
    const offset = (time * .015) % spacing;
    for (let x = -spacing + offset; x < width + spacing; x += spacing) {
      closingCtx.beginPath();
      closingCtx.moveTo(x, 0);
      closingCtx.lineTo(x + height * .3, height);
      closingCtx.stroke();
    }
    for (let y = -spacing + offset; y < height + spacing; y += spacing) {
      closingCtx.beginPath();
      closingCtx.moveTo(0, y);
      closingCtx.lineTo(width, y + width * .08);
      closingCtx.stroke();
    }
    requestAnimationFrame(drawClosing);
  };
  resizeClosing();
  drawClosing();
  window.addEventListener('resize', resizeClosing);

  // ─── TOAST ──────────────────────────────────────────────────────────────────
  const demoToast = $('#demoToast');
  $('#closeToast').addEventListener('click', () => {
    demoToast.animate([
      { transform: 'translateY(0)', opacity: 1 },
      { transform: 'translateY(140%)', opacity: 0 }
    ], { duration: 350, fill: 'forwards', easing: 'ease-in' });
  });

  // ─── LIVE DATA INTEGRATION ──────────────────────────────────────────────────

  /** Update city-wide pulse metrics from /hotspots */
  async function loadPulseMetrics() {
    try {
      const data = await apiFetch('/hotspots?limit=100');
      const hotspots = Array.isArray(data) ? data : (data.hotspots ?? data.results ?? []);
      state.hotspots = hotspots;

      const totalHotspots = hotspots.length;
      const structural = hotspots.filter(h => (h.persistence_score ?? h.cluster_persistence_score ?? 0) > 0.7).length;
      const avgRisk = hotspots.length
        ? Math.round(hotspots.reduce((s, h) => s + (h.risk_score ?? h.violation_count ?? 50), 0) / hotspots.length)
        : 72;
      const topHotspot = hotspots[0];

      // Update DOM
      const cityRiskEl = $('#cityRiskMetric');
      const heroRiskEl = $('#heroRisk');
      if (cityRiskEl) animateNumber(cityRiskEl, avgRisk, 0);
      if (heroRiskEl) animateNumber(heroRiskEl, avgRisk, 0);
      state.cityRisk = avgRisk;

      const hotspotMetric = $('#hotspotMetric');
      if (hotspotMetric) animateNumber(hotspotMetric, totalHotspots, 0);

      const structuralEl = hotspotMetric?.closest('.metric-pair')?.querySelector('div:last-child strong');
      if (structuralEl) animateNumber(structuralEl, structural, 0);

      // Top hotspot name
      const placeEl = $('.metric-place');
      if (placeEl && topHotspot) {
        placeEl.textContent = topHotspot.zone_name ?? topHotspot.junction_name ?? `Cluster ${topHotspot.cluster_id}`;
      }

      // Signal bar
      const signalBar = $('.signal-bar span');
      if (signalBar) signalBar.style.setProperty('--signal', `${avgRisk}%`);

      // Intelligence card — top zone prediction
      const riskScoreEl = $('.int-card-primary .int-card-copy strong');
      if (riskScoreEl && topHotspot) animateNumber(riskScoreEl, topHotspot.risk_score ?? topHotspot.violation_count ?? 86, 0);
      const riskZoneEl = $('.int-card-primary .int-card-copy p');
      if (riskZoneEl && topHotspot) riskZoneEl.textContent = (topHotspot.zone_name ?? `Cluster ${topHotspot.cluster_id}`) + ' · Evening shift';

      // Map focus card critical corridor
      const corridorEl = $('.map-focus-card strong');
      if (corridorEl && hotspots.length >= 2) {
        const n0 = hotspots[0].zone_name ?? `Cluster ${hotspots[0].cluster_id}`;
        const n1 = hotspots[1].zone_name ?? `Cluster ${hotspots[1].cluster_id}`;
        corridorEl.textContent = `${n0} → ${n1}`;
      }

      // Rebuild network nodes from real data
      makeNetwork();

      // Swap toast message to live data confirmed
      const toastP = demoToast?.querySelector('p');
      if (toastP) toastP.innerHTML = '<b>Live mode</b> — connected to BTIP backend at localhost:8000.';

    } catch (err) {
      console.warn('Hotspots API unavailable — using mock data:', err);
    }
  }

  /** Update SHAP reasons in intelligence card B from /risk */
  async function loadRiskReasons() {
    try {
      const today = new Date().toISOString().split('T')[0];
      const data = await apiFetch(`/risk?zone_id=1&shift=Evening&date=${today}`);
      const shap = data.shap_explanations ?? [];
      const reasonList = $('.reason-list');
      if (!reasonList || !shap.length) return;
      reasonList.innerHTML = shap.slice(0, 3).map(item => {
        const sign = item.direction === '+' || item.impact > 0 ? '+' : '';
        const val = typeof item.impact === 'number' ? Math.abs(Math.round(item.impact)) : Math.abs(item.value ?? 0).toFixed(1);
        return `<li><span>${item.feature}</span><b>${sign}${val}</b></li>`;
      }).join('');

      // Confidence pill
      const confPill = $('.confidence-pill');
      if (confPill && data.confidence_band) {
        const p50 = data.confidence_band.p50 ?? data.risk_score;
        confPill.textContent = `${Math.round(p50)}% confidence`;
      } else if (confPill && data.risk_score) {
        confPill.textContent = `${Math.round(data.risk_score)}% risk score`;
      }
    } catch (err) {
      console.warn('Risk API unavailable:', err);
    }
  }

  /** Update recommendations section from /recommendations */
  async function loadRecommendations() {
    try {
      const today = new Date().toISOString().split('T')[0];
      const data = await apiFetch(`/recommendations?shift=Evening&date=${today}&total_officers=20`);
      const recs = Array.isArray(data) ? data : (data.recommendations ?? data.results ?? []);
      state.recommendations = recs;
      if (!recs.length) return;

      // Update orbit center
      const orbitCenter = $('.orbit-center');
      const totalOfficers = recs.reduce((s, r) => s + (r.n_officers ?? r.recommended_officers ?? 0), 0);
      if (orbitCenter) {
        orbitCenter.querySelector('strong').textContent = `${recs.length} zones`;
        orbitCenter.querySelector('small').textContent = `${totalOfficers} OFFICERS`;
      }

      // Update orbit nodes
      const orbitNodes = $$('.orbit-node');
      recs.slice(0, orbitNodes.length).forEach((rec, i) => {
        const node = orbitNodes[i];
        if (!node) return;
        const officers = rec.n_officers ?? rec.recommended_officers ?? 0;
        const name = rec.zone_name ?? rec.junction_name ?? `Zone ${rec.zone_id}`;
        node.querySelector('b').textContent = officers;
        node.querySelector('span').textContent = name.split(' ').slice(0, 2).join(' ');
      });

      // Update recommendation cards
      const cards = $$('.recommendation-card');
      recs.slice(0, cards.length).forEach((rec, i) => {
        const card = cards[i];
        if (!card) return;
        const name = rec.zone_name ?? rec.junction_name ?? `Zone ${rec.zone_id}`;
        const officers = rec.n_officers ?? rec.recommended_officers ?? 0;
        const riskBefore = rec.risk_score ?? 80;
        const riskAfter = Math.round(riskBefore * (1 - (rec.expected_reduction_pct ?? 30) / 100));
        const shift = rec.recommended_shift ?? 'Evening';

        card.querySelector('h3').textContent = name;
        card.querySelector('.rec-main p').textContent = `Deploy ${officers} officers · ${shift}.`;
        const impact = card.querySelector('.rec-impact strong');
        if (impact) impact.textContent = `${riskBefore} → ${riskAfter}`;

        // SHAP explanations in the accordion
        const detail = card.querySelector('.rec-detail');
        if (detail && rec.shap_explanations?.length) {
          const chips = rec.shap_explanations.slice(0, 3)
            .map(s => `<span style="display:inline-flex;gap:4px;align-items:center;background:rgba(0,0,0,.06);border-radius:4px;padding:2px 8px;font-size:.78rem;margin:2px">${s.feature} <b style="color:${s.direction === '+' ? '#e8530a' : '#2f73ff'}">${s.direction}${Math.abs(s.impact ?? 0).toFixed(0)}</b></span>`)
            .join('');
          detail.innerHTML = `<p style="margin-bottom:8px">${rec.explanation?.top_drivers?.[0] ?? 'Multiple risk factors converge here.'}</p><div style="display:flex;flex-wrap:wrap;gap:4px">${chips}</div>`;
        }
      });

    } catch (err) {
      console.warn('Recommendations API unavailable:', err);
    }
  }

  /** Poll every 60s for city pulse refresh */
  function startPolling() {
    setInterval(loadPulseMetrics, 60_000);
  }

  // ─── BOOT SEQUENCE ──────────────────────────────────────────────────────────
  (async () => {
    await loadPulseMetrics();
    await Promise.allSettled([loadRiskReasons(), loadRecommendations()]);
    startPolling();
  })();

})();
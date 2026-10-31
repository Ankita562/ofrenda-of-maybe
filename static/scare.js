() => {
  if (window.__ofrenda) return;
  window.__ofrenda = true;

  // Full-screen ghost overlay (styled in style.css)
  const ov = document.createElement('div');
  ov.id = 'scare';
  ov.innerHTML = '<span>👻</span>';
  document.body.appendChild(ov);

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Read the Settings checkboxes (elem_id: scare_toggle, sound_toggle)
  const on = id => {
    const el = document.querySelector('#' + id + ' input[type=checkbox]');
    return el ? el.checked : true;
  };

  // Scream made with the Web Audio API, so no audio file is needed
  const scream = () => {
    try {
      const c = new (window.AudioContext || window.webkitAudioContext)();
      const o = c.createOscillator();
      const g = c.createGain();
      o.type = 'sawtooth';
      o.frequency.setValueAtTime(900, c.currentTime);
      o.frequency.exponentialRampToValueAtTime(90, c.currentTime + 0.6);
      g.gain.setValueAtTime(0.25, c.currentTime);
      g.gain.exponentialRampToValueAtTime(0.001, c.currentTime + 0.7);
      o.connect(g);
      g.connect(c.destination);
      o.start();
      o.stop(c.currentTime + 0.7);
    } catch (e) {}
  };

  const scare = () => {
    ov.classList.add('on');
    if (on('sound_toggle')) scream();
    setTimeout(() => ov.classList.remove('on'), 900);
  };

  // First scare after ~20 keystrokes, then rarely (25 s cooldown)
  let typed = 0, first = false, last = 0;
  document.addEventListener('input', e => {
    if (!e.target.closest || !e.target.closest('#wish')) return;
    if (reduce || !on('scare_toggle')) return;
    typed++;
    const now = Date.now();
    const due = (!first && typed >= 20) ||
                (first && typed >= 12 && now - last > 25000 && Math.random() < 0.08);
    if (due) { first = true; last = now; typed = 0; scare(); }
  });
}
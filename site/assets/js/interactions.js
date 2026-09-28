/* ============================================================
   AI SYSTEMS PORTFOLIO — Interactions
   Scroll reveal, counter animations, rotating hero text
   ============================================================ */

(function () {
  'use strict';

  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ── Scroll reveal ─────────────────────────────────────────── */
  const revealEls = document.querySelectorAll('.reveal');

  if (revealEls.length) {
    if (prefersReducedMotion) {
      revealEls.forEach(el => el.classList.add('visible'));
    } else {
      const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            revealObserver.unobserve(entry.target);
          }
        });
      }, { threshold: 0.1 });

      revealEls.forEach(el => revealObserver.observe(el));
    }
  }

  /* ── Counter animations ────────────────────────────────────── */
  function animateCounter(el) {
    const target = parseInt(el.dataset.target, 10);
    const suffix = el.dataset.suffix || '';

    if (prefersReducedMotion) {
      el.textContent = target.toLocaleString() + suffix;
      return;
    }

    const duration = 1400;
    const start = performance.now();

    function update(now) {
      const elapsed  = Math.min(now - start, duration);
      const progress = elapsed / duration;
      const eased    = 1 - Math.pow(1 - progress, 3);
      const value    = Math.round(eased * target);

      el.textContent = value.toLocaleString() + suffix;

      if (elapsed < duration) {
        requestAnimationFrame(update);
      } else {
        el.textContent = target.toLocaleString() + suffix;
      }
    }

    requestAnimationFrame(update);
  }

  const counterEls = document.querySelectorAll('.stat-number[data-target]');

  if (counterEls.length) {
    const counterObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          counterObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });

    counterEls.forEach(el => counterObserver.observe(el));
  }

  /* ── Rotating hero text ────────────────────────────────────── */
  const rotatingEl = document.getElementById('rotatingText');

  if (rotatingEl && !prefersReducedMotion) {
    const phrases = [
      'that ship to production.',
      'for agents and platforms.',
      'with structure and discipline.',
    ];
    let currentIndex = 0;

    function rotatePhrases() {
      rotatingEl.style.transition = 'opacity 0.4s ease';
      rotatingEl.style.opacity    = '0';

      setTimeout(() => {
        currentIndex = (currentIndex + 1) % phrases.length;
        rotatingEl.textContent  = phrases[currentIndex];
        rotatingEl.style.opacity = '1';
      }, 420);
    }

    let rotationCount  = 0;
    const maxRotations = phrases.length * 2;

    const interval = setInterval(() => {
      rotatePhrases();
      rotationCount++;
      if (rotationCount >= maxRotations) {
        clearInterval(interval);
        setTimeout(() => {
          rotatingEl.style.opacity = '0';
          setTimeout(() => {
            rotatingEl.textContent  = 'for production AI systems.';
            rotatingEl.style.opacity = '1';
          }, 420);
        }, 1000);
      }
    }, 3500);
  }

})();

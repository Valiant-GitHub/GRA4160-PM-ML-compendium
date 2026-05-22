/* GRA 4160 study companion — global math-mode toggle (C9).
 *
 * Persists to localStorage key `gra4160.mathMode` ∈ {off, course, full}.
 * Drives <body> classes math-off | math-a | math-b, which the SCSS uses to
 * show/hide .math-mode-a / .math-mode-b regions site-wide WITHOUT page reload.
 * Default on first visit: off (both math modes collapsed).
 *
 * This file is the canonical source. It is injected on every page via
 * `include-after-body: theme/toggle-after-body.html` in _quarto.yml, which
 * inlines an identical copy inside <script> tags (kept in sync with this file).
 */
(function () {
  "use strict";
  var KEY = "gra4160.mathMode";
  var ORDER = ["off", "course", "full"];
  var BODYCLASS = { off: "math-off", course: "math-a", full: "math-b" };
  var LABEL = { off: "off", course: "course level", full: "full" };

  function read() {
    var v = null;
    try { v = window.localStorage.getItem(KEY); } catch (e) { /* private mode */ }
    return ORDER.indexOf(v) >= 0 ? v : "off";
  }
  function write(v) { try { window.localStorage.setItem(KEY, v); } catch (e) {} }

  function apply(mode) {
    var b = document.body;
    if (!b) return;
    b.classList.remove("math-off", "math-a", "math-b");
    b.classList.add(BODYCLASS[mode]);
    var st = document.querySelector("#math-mode-toggle .mm-state");
    if (st) st.textContent = LABEL[mode];
    var btn = document.getElementById("math-mode-toggle");
    if (btn) btn.setAttribute("aria-label", "Math display: " + LABEL[mode] + " (click to change)");
  }

  function next(mode) { return ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length]; }

  function makeButton() {
    if (document.getElementById("math-mode-toggle")) return;
    var btn = document.createElement("button");
    btn.id = "math-mode-toggle";
    btn.type = "button";
    btn.title = "Cycle math display: off → course level → full";
    btn.innerHTML = 'Math: <span class="mm-state">off</span>';
    btn.addEventListener("click", function () {
      var m = next(read());
      write(m);
      apply(m);
    });

    // Prefer the Quarto navbar; fall back to a fixed floating control.
    var host = document.querySelector(".navbar .navbar-nav.navbar-nav-scroll")
            || document.querySelector("#quarto-search ~ *")
            || document.querySelector(".navbar .quarto-navbar-tools")
            || document.querySelector(".navbar");
    if (host && host.classList.contains("navbar")) {
      var wrap = document.createElement("div");
      wrap.className = "navbar-nav ms-auto";
      wrap.appendChild(btn);
      host.appendChild(wrap);
    } else if (host) {
      host.appendChild(btn);
    } else {
      btn.style.position = "fixed";
      btn.style.top = "0.5rem";
      btn.style.right = "0.75rem";
      btn.style.zIndex = "1080";
      document.body.appendChild(btn);
    }
  }

  function init() {
    makeButton();
    apply(read());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

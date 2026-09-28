// Client-side conveniences only. Every check here is repeated on the server.
(function () {
  "use strict";

  // Probability bars: widths come from data attributes (inline styles are blocked by the CSP).
  document.querySelectorAll(".bar-fill[data-w]").forEach(function (el) {
    var w = parseFloat(el.getAttribute("data-w"));
    if (isFinite(w)) el.style.width = Math.max(0, Math.min(100, w)) + "%";
  });

  var form = document.getElementById("upload-form");
  if (!form) return;
  var audioMax = +form.dataset.audioMax, imageMax = +form.dataset.imageMax;
  var minS = +form.dataset.minSeconds, maxS = +form.dataset.maxSeconds;
  var audio = document.getElementById("audio"), image = document.getElementById("image");
  var aStatus = document.getElementById("audio-status"), iStatus = document.getElementById("image-status");
  var aPrev = document.getElementById("audio-preview"), iPrev = document.getElementById("image-preview");
  var problems = { audio: "", image: "" };

  function mb(n) { return (n / 1048576).toFixed(1) + " MB"; }
  function set(el, msg, bad) { el.textContent = msg; el.className = bad ? "help error" : "help"; }

  audio.addEventListener("change", function () {
    problems.audio = "";
    aPrev.hidden = true;
    var f = audio.files[0];
    if (!f) { set(aStatus, ""); return; }
    if (f.size > audioMax) { problems.audio = "Audio file is " + mb(f.size) + "; the limit is " + mb(audioMax) + "."; set(aStatus, problems.audio, true); return; }
    if (f.type && f.type.indexOf("audio/") !== 0 && f.type !== "video/webm") { problems.audio = "This does not look like an audio file."; set(aStatus, problems.audio, true); return; }
    var url = URL.createObjectURL(f);
    aPrev.src = url;
    aPrev.hidden = false;
    aPrev.onloadedmetadata = function () {
      var d = aPrev.duration;
      if (isFinite(d) && (d < minS || d > maxS)) { problems.audio = "Recording is " + d.toFixed(1) + " s; it must be " + minS + "–" + maxS + " s."; set(aStatus, problems.audio, true); }
      else set(aStatus, f.name + " · " + mb(f.size) + (isFinite(d) ? " · " + d.toFixed(1) + " s" : ""));
    };
    aPrev.onerror = function () { set(aStatus, "Your browser cannot preview this file; the server will still check it.", false); };
  });

  image.addEventListener("change", function () {
    problems.image = "";
    iPrev.hidden = true;
    var f = image.files[0];
    if (!f) { set(iStatus, ""); return; }
    if (f.size > imageMax) { problems.image = "Image is " + mb(f.size) + "; the limit is " + mb(imageMax) + "."; set(iStatus, problems.image, true); return; }
    if (["image/jpeg", "image/png", "image/webp"].indexOf(f.type) < 0) { problems.image = "Use a JPEG, PNG or WebP image."; set(iStatus, problems.image, true); return; }
    iPrev.src = URL.createObjectURL(f);
    iPrev.hidden = false;
    iPrev.onload = function () { set(iStatus, f.name + " · " + mb(f.size) + " · " + iPrev.naturalWidth + "×" + iPrev.naturalHeight); };
    iPrev.onerror = function () { problems.image = "The image could not be read — it may be damaged."; set(iStatus, problems.image, true); iPrev.hidden = true; };
  });

  form.addEventListener("submit", function (e) {
    var msg = problems.audio || problems.image;
    if (!audio.files[0] && !image.files[0]) msg = "Choose an audio recording, an image, or both.";
    if (msg) { e.preventDefault(); (problems.audio ? aStatus : iStatus).focus(); set(problems.audio ? aStatus : iStatus, msg, true); return; }
    document.getElementById("submit-btn").disabled = true;
    document.getElementById("busy").hidden = false;
  });
})();

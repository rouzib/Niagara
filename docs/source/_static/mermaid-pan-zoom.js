(() => {
  "use strict";

  const views = new WeakMap();
  const selector = ".mermaid svg, .mermaid-container-fullscreen svg";
  let pending = false;

  function boxFrom(text) {
    const numbers = (text || "").trim().split(/[ ,]+/).map(Number);
    return numbers.length === 4 && numbers.every(Number.isFinite) &&
      numbers[2] > 0 && numbers[3] > 0 ? numbers : null;
  }

  function viewport(svg) {
    const saved = boxFrom(svg.getAttribute("viewBox"));
    if (saved) return saved;
    try {
      const box = svg.getBBox();
      return box.width > 0 && box.height > 0 ? [box.x, box.y, box.width, box.height] : null;
    } catch { return null; }
  }

  class DiagramView {
    constructor(svg) {
      this.svg = svg;
      const natural = viewport(svg);
      if (!natural) throw new Error("Diagram has no usable viewport");
      this.initial = boxFrom(svg.dataset.niagaraOriginalViewBox) || natural;
      this.box = natural.slice();
      svg.setAttribute("viewBox", this.box.join(" "));
      this.drag = null;
      svg.dataset.niagaraOriginalViewBox = this.initial.join(" ");
      svg.dataset.niagaraPanZoom = "true";
      if (!svg.hasAttribute("tabindex")) svg.setAttribute("tabindex", "0");
      if (!svg.hasAttribute("aria-label") && !svg.hasAttribute("aria-labelledby")) {
        svg.setAttribute("aria-label", "Interactive diagram");
      }
      svg.setAttribute("aria-description", "Drag or use arrow keys to pan. Use plus, minus, or the wheel to zoom. Press zero to reset.");
      svg.style.touchAction = "none";
      svg.style.userSelect = "none";
      svg.style.cursor = "grab";
      svg.addEventListener("wheel", (event) => {
        if (event.ctrlKey || event.metaKey) return;
        event.preventDefault();
        const delta = Math.max(-160, Math.min(160, event.deltaY));
        this.zoom(Math.exp(-delta * 0.004), this.point(event));
      }, { passive: false });
      svg.addEventListener("keydown", (event) => this.key(event));
      svg.addEventListener("pointerdown", (event) => this.begin(event));
      svg.addEventListener("pointermove", (event) => this.move(event));
      for (const kind of ["pointerup", "pointercancel", "lostpointercapture"]) {
        svg.addEventListener(kind, (event) => {
          if (this.drag && this.drag.id === event.pointerId) this.stop();
        });
      }
      this.controls();
    }

    point(event, inverse) {
      const matrix = inverse || this.svg.getScreenCTM()?.inverse();
      if (!matrix) return null;
      const point = this.svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      return point.matrixTransform(matrix);
    }

    write(box) {
      this.box = box;
      this.svg.setAttribute("viewBox", box.join(" "));
    }

    zoom(factor, point) {
      this.stop();
      const scale = this.initial[2] / this.box[2];
      const next = Math.min(12, Math.max(0.2, scale * factor));
      const ratio = scale / next;
      const [x, y, width, height] = this.box;
      const anchor = point || { x: x + width / 2, y: y + height / 2 };
      this.write([
        anchor.x + (x - anchor.x) * ratio,
        anchor.y + (y - anchor.y) * ratio,
        width * ratio, height * ratio,
      ]);
    }

    reset() {
      this.stop();
      this.write(this.initial.slice());
    }

    begin(event) {
      if (event.button !== 0 || event.target.closest("a")) return;
      const inverse = this.svg.getScreenCTM()?.inverse();
      if (!inverse) return;
      event.preventDefault();
      this.svg.focus({ preventScroll: true });
      this.drag = { id: event.pointerId, inverse, start: this.point(event, inverse), box: this.box.slice() };
      this.svg.setPointerCapture(event.pointerId);
      this.svg.classList.add("is-panning");
      this.svg.style.cursor = "grabbing";
    }

    move(event) {
      if (!this.drag || this.drag.id !== event.pointerId) return;
      // A fixed screen transform keeps drag distances stable while viewBox changes.
      const point = this.point(event, this.drag.inverse);
      const [x, y, width, height] = this.drag.box;
      this.write([x + this.drag.start.x - point.x, y + this.drag.start.y - point.y, width, height]);
    }

    stop() {
      const active = this.drag;
      this.drag = null;
      if (active && this.svg.hasPointerCapture(active.id)) this.svg.releasePointerCapture(active.id);
      this.svg.classList.remove("is-panning");
      this.svg.style.cursor = "grab";
    }

    key(event) {
      if (event.target !== this.svg || event.ctrlKey || event.metaKey || event.altKey) return;
      const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      if (directions[event.key]) {
        const [dx, dy] = directions[event.key];
        const [x, y, width, height] = this.box;
        this.stop();
        this.write([x + dx * width * 0.08, y + dy * height * 0.08, width, height]);
      } else if (event.key === "+" || event.key === "=") this.zoom(1.25);
      else if (event.key === "-") this.zoom(0.8);
      else if (event.key === "0") this.reset();
      else return;
      event.preventDefault();
    }

    controls() {
      const host = this.svg.closest(".mermaid-container-fullscreen") || this.svg.closest(".mermaid-container");
      if (!host || (this.toolbar?.isConnected && this.toolbar.parentElement === host)) return;
      this.toolbar?.remove();
      for (const old of host.querySelectorAll(":scope > .niagara-mermaid-controls")) old.remove();
      const toolbar = document.createElement("div");
      toolbar.className = "niagara-mermaid-controls";
      toolbar.setAttribute("role", "group");
      toolbar.setAttribute("aria-label", "Diagram pan and zoom controls");
      for (const [label, title, action] of [
        ["+", "Zoom in", () => this.zoom(1.25)],
        ["-", "Zoom out", () => this.zoom(0.8)],
        ["0", "Reset diagram view", () => this.reset()],
      ]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.title = title;
        button.setAttribute("aria-label", title);
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          action();
          this.svg.focus({ preventScroll: true });
        });
        toolbar.appendChild(button);
      }
      host.appendChild(toolbar);
      this.toolbar = toolbar;
    }
  }

  function refresh() {
    pending = false;
    for (const svg of document.querySelectorAll(selector)) {
      if (views.has(svg)) views.get(svg).controls();
      else if (viewport(svg)) views.set(svg, new DiagramView(svg));
    }
  }

  function schedule() {
    if (!pending) {
      pending = true;
      window.requestAnimationFrame(refresh);
    }
  }

  function start() {
    schedule();
    // Rendering, theme changes and fullscreen clones can replace diagram elements.
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();

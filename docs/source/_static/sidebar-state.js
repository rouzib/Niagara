(() => {
  "use strict";

  function start() {
    const nav = document.querySelector(".bd-sidebar-primary nav.bd-docs-nav");
    if (!nav) return;
    const key = "niagara-sidebar-tree-state-v1";
    const state = new Map();
    try {
      const stored = JSON.parse(localStorage.getItem(key) || "{}");
      if (stored && typeof stored === "object" && !Array.isArray(stored)) {
        for (const [path, open] of Object.entries(stored)) {
          if (typeof open === "boolean") state.set(path, open);
        }
      }
    } catch { /* Native navigation remains available without storage. */ }

    const save = () => {
      try { localStorage.setItem(key, JSON.stringify(Object.fromEntries(state))); }
      catch { /* Storage may be disabled for a local documentation file. */ }
    };
    const branches = new Map();
    for (const details of nav.querySelectorAll("li.has-children > details")) {
      const item = details.parentElement;
      const link = Array.from(item.children).find((child) => child.matches("a.reference.internal"));
      if (!link) continue;
      let path;
      try {
        path = new URL(link.getAttribute("href"), document.baseURI).pathname;
        path = path.replace(/\/index\.html$/, "/").replace(/\/+$/, "") || "/";
      } catch { continue; }
      branches.set(details, path);
      if (state.has(path)) details.open = state.get(path);
    }

    // Toggle does not bubble, so capture it at the navigation root.
    nav.addEventListener("toggle", (event) => {
      const path = branches.get(event.target);
      if (path !== undefined) {
        state.set(path, event.target.open);
        save();
      }
    }, true);
    nav.addEventListener("click", (event) => {
      const link = event.target.closest("a.reference.internal");
      if (!link || !nav.contains(link)) return;
      for (let item = link.closest("li"); item && nav.contains(item); item = item.parentElement?.closest("li")) {
        const details = Array.from(item.children).find((child) => child.matches("details"));
        const path = branches.get(details);
        if (path !== undefined) {
          details.open = true;
          state.set(path, true);
        }
      }
      save();
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();

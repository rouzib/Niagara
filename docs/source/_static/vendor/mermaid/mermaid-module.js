const bundle = new URL("mermaid.min.js?release=11.12.1", import.meta.url);

async function diagramRuntime() {
  if (!globalThis.mermaid) {
    const loader = document.createElement("script");
    loader.async = true;
    loader.src = bundle.href;
    await new Promise((loaded, failed) => {
      loader.addEventListener("load", loaded, { once: true });
      loader.addEventListener("error", () => {
        loader.remove();
        failed(new Error("Could not load the bundled diagram renderer"));
      }, { once: true });
      document.head.appendChild(loader);
    });
  }
  const runtime = globalThis.mermaid;
  if (!runtime || typeof runtime.initialize !== "function" || typeof runtime.run !== "function") {
    throw new Error("The bundled diagram renderer has no browser API");
  }
  return runtime;
}

export default await diagramRuntime();

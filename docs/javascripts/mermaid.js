window.addEventListener("load", async () => {
  const diagrams = [...document.querySelectorAll(".mermaid")];
  diagrams.forEach((diagram) => {
    diagram.dataset.mermaidSource = diagram.textContent.trim();
  });

  const scheme = () => document.body.getAttribute("data-md-color-scheme");
  let renderedScheme = scheme();
  let rendering = Promise.resolve();

  const render = () => {
    rendering = rendering.then(async () => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: scheme() === "slate" ? "dark" : "default",
      });
      diagrams.forEach((diagram) => {
        diagram.removeAttribute("data-processed");
        diagram.textContent = diagram.dataset.mermaidSource;
      });
      await mermaid.run({ nodes: diagrams });
      renderedScheme = scheme();
    });
  };

  render();
  new MutationObserver(() => {
    if (scheme() !== renderedScheme) render();
  }).observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
});

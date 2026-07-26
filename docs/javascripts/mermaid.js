window.addEventListener("load", () => {
  mermaid.initialize({
    startOnLoad: true,
    theme: document.body.getAttribute("data-md-color-scheme") === "slate"
      ? "dark"
      : "default",
  });
});

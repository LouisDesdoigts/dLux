window.addEventListener("load", () => {
  mermaid.initialize({
    startOnLoad: true,
    securityLevel: "loose",
    theme: document.body.getAttribute("data-md-color-scheme") === "slate"
      ? "dark"
      : "default",
  });
});

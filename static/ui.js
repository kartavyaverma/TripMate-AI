// Presentation-only: mirrors the existing section visibility into the itinerary flow bar.
// Does not call or modify any of the planner logic in script.js.
(function () {
  const tabs = document.querySelectorAll(".flow-tab");
  if (!tabs.length) {
    return;
  }

  const byId = (id) => document.getElementById(id);
  const isVisible = (el) => el && !el.classList.contains("hidden");

  const STEP_TARGETS = {
    plan: "top",
    live: "workflowSection",
    review: "approvalSection",
    itinerary: "resultSection"
  };

  function currentStep() {
    const sendBtn = byId("sendBtn");
    const resultTitle = byId("resultTitle");

    if (isVisible(byId("approvalSection"))) {
      return "review";
    }
    if (sendBtn && sendBtn.disabled) {
      return "live";
    }
    if (isVisible(byId("resultSection")) && resultTitle && /final/i.test(resultTitle.textContent)) {
      return "itinerary";
    }
    return "plan";
  }

  function render() {
    const active = currentStep();
    tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.step === active));
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      let target = byId(STEP_TARGETS[tab.dataset.step]);
      if (!isVisible(target)) {
        target = byId("top");
      }
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  const observer = new MutationObserver(render);
  ["sendBtn", "approvalSection", "resultSection", "resultTitle"].forEach((id) => {
    const el = byId(id);
    if (el) {
      observer.observe(el, { attributes: true, childList: true, characterData: true, subtree: true });
    }
  });

  render();
})();

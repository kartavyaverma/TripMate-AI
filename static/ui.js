// Presentation-only: reflects the app's real state in the "How it's planned" steps.
// It only reads what script.js renders; it never calls the planner logic.
(function () {
  const list = document.getElementById("planSteps");
  if (!list) {
    return;
  }

  const byId = (id) => document.getElementById(id);
  const isVisible = (el) => el && !el.classList.contains("hidden");
  const steps = {};
  list.querySelectorAll("li").forEach((li) => {
    steps[li.dataset.step] = li;
  });

  const STATES = ["is-active", "is-working", "is-done", "is-skipped", "is-stopped"];
  const ORDER = ["check", "gather", "budget", "draft", "review", "final"];

  function set(step, state) {
    const li = steps[step];
    li.classList.remove(...STATES);
    if (state) {
      li.classList.add(state);
    }
  }

  function ranAgents() {
    return new Set(
      Array.from(document.querySelectorAll("#agentChips .agent-chip")).map((chip) => chip.dataset.agent)
    );
  }

  // Steps 1-4 after a draft or final plan exists: tick what actually ran.
  function markPlanned() {
    const agents = ranAgents();
    const gathered = ["flight_agent", "hotel_agent", "weather_agent"].some((a) => agents.has(a));
    set("check", "is-done");
    set("gather", gathered ? "is-done" : "is-skipped");
    set("budget", agents.has("budget_agent") ? "is-done" : "is-skipped");
    set("draft", "is-done");
  }

  function render() {
    const sendBtn = byId("sendBtn");
    const approveBtn = byId("approveBtn");
    const result = byId("resultSection");
    const note = byId("stepsNote");
    const kind = isVisible(result) ? result.dataset.kind : "";
    const reviewing = isVisible(byId("approvalSection"));

    ORDER.forEach((step) => set(step, ""));
    note.textContent = "";

    if (reviewing && approveBtn.disabled) {
      markPlanned();
      set("review", "is-done");
      set("final", "is-working");
      note.textContent = "Writing the final plan…";
    } else if (reviewing) {
      markPlanned();
      set("review", "is-active");
      note.textContent = "Waiting for you";
    } else if (sendBtn.disabled) {
      ["check", "gather", "budget", "draft"].forEach((step) => set(step, "is-working"));
      note.textContent = "Usually 30–40 seconds";
    } else if (kind === "blocked") {
      set("check", "is-stopped");
      note.textContent = "Stopped at the guardrail";
    } else if (kind === "final") {
      markPlanned();
      set("review", "is-done");
      set("final", "is-done");
      note.textContent = "Done";
    }
  }

  const observer = new MutationObserver(render);
  ["sendBtn", "approveBtn", "approvalSection", "resultSection", "agentChips"].forEach((id) => {
    const el = byId(id);
    if (el) {
      observer.observe(el, { attributes: true, childList: true });
    }
  });

  render();
})();

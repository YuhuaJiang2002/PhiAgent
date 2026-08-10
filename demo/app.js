"use strict";

const actions = {
  slideRight: {
    title: "LIFT-ARC CARRY RIGHT / NATIVE / ACCEPTED",
    label: "OSCAR NATIVE · SHOULDER + ELBOW + WRIST + FINGER ARTICULATION",
    video: "showcase/oscar-acwm-carry-right.mp4?v=7",
    prompt: "用右侧机器人手抓住黄色碗的把手，先轻轻抬离桌面，再把碗搬到画面右侧目标区并保持。",
    timeline: "0–0.8s 接近把手；0.8–1.3s 建立抓取；1.3–3.6s 轻抬并沿弧线搬到右侧；3.6–5.2s 在右侧终点保持。",
    state: ["OSCAR NATIVE / LIFT-ARC CARRY RIGHT / ACCEPTED", "参考上提视频的原生生成路径重新编译竖直模板和右向弧线；手臂、手腕与手指连续运动，黄碗保持接触并通过自动门与人工复核。", "complete"],
  },
  liftUp: {
    title: "LIFT UP / ACCEPTED",
    label: "ACTION 03 · CONTACT → LIFT UP → HOLD",
    video: "showcase/oscar-acwm-lift-up.mp4?v=5",
    prompt: "用右侧机器人手接触黄色碗，把碗明显向画面上方提起并保持。",
    timeline: "0–1.0s 接近并建立接触；1.0–3.7s 向画面上方持续提起；3.7–5.4s 在上方保持。",
    state: ["OSCAR / LIFT UP / ACCEPTED", "动作、机器人形态、物体交互、时序、背景与人工复核全部通过。", "complete"],
  },
  slideLeft: {
    title: "SLIDE LEFT / REJECTED",
    label: "ACTION 01 · ROBOT RETRACTS · OBJECT DOES NOT FOLLOW",
    video: "showcase/oscar-acwm-slide-left-rejected.mp4?v=5",
    prompt: "用右侧机器人手接触黄色碗，把碗沿桌面明显移动到画面左侧，然后保持。",
    timeline: "0–1.0s 接近并建立接触；1.0–3.8s 持续向画面左侧移动；3.8–5.4s 在左侧终点保持。",
    state: ["OSCAR / SLIDE LEFT / REJECTED", "机器人产生左移，但黄色碗没有形成因果运动；增强末端条件后的 repair 重跑仍失败。", "error"],
  },
};

const views = {
  comparison: {
    title: "ACCEPTED COUNTERFACTUALS",
    label: "NATIVE LIFT-ARC CARRY RIGHT vs NATIVE LIFT · BOTH ACCEPTED",
    video: "showcase/oscar-acwm-accepted-comparison.mp4?v=7",
  },
  repairComparison: {
    title: "HISTORICAL SLIDE RIGHT / TWO REJECTED ATTEMPTS",
    label: "HAND DRIFT vs RIGID WHOLE-HAND SHIFT · BOTH USER REJECTED",
    video: "showcase/oscar-acwm-slide-right-raw-vs-structure-lock.mp4?v=7",
  },
  slideRightRaw: {
    title: "SLIDE RIGHT / RAW OSCAR / USER REJECTED",
    label: "LATE-FRAME FINGER TOPOLOGY FRAGMENTATION",
    video: "showcase/oscar-acwm-slide-right-raw.mp4?v=7",
  },
  source: {
    title: "HAND2DEX-2 REAL-SCENE SOURCE",
    label: "ORIGINAL REAL VIDEO · SHARED FIRST FRAME",
    video: "showcase/oscar-acwm-real-source.mp4?v=5",
  },
  rightCondition: {
    title: "LIFT-ARC CARRY RIGHT / ACTION CONDITION",
    label: "CAMERA:SKELETON · LIFT TEMPLATE + RIGHTWARD ARC · 81 FRAMES",
    video: "showcase/oscar-acwm-carry-right-condition.mp4?v=7",
  },
  liftCondition: {
    title: "LIFT UP / ACTION CONDITION",
    label: "CAMERA:SKELETON · 81 FRAMES",
    video: "showcase/oscar-acwm-lift-up-condition.mp4?v=5",
  },
  ...actions,
};

const mainVideo = document.querySelector("#mainVideo");
const commandInput = document.querySelector("#commandInput");
const timelineInput = document.querySelector("#timelineInput");
const form = document.querySelector("#commandForm");
const jobState = document.querySelector("#jobState");
const jobTitle = document.querySelector("#jobTitle");
const jobDetail = document.querySelector("#jobDetail");
const charCount = document.querySelector("#charCount");

function setState(title, detail, mode) {
  jobTitle.textContent = title;
  jobDetail.textContent = detail;
  jobState.dataset.mode = mode;
}

function updateCount() {
  charCount.textContent = `${commandInput.value.length} / 700`;
}

function loadView(key, autoplay = true) {
  const view = views[key];
  if (!view) return;
  mainVideo.pause();
  mainVideo.src = view.video;
  mainVideo.load();
  document.querySelector("#viewTitle").textContent = view.title;
  document.querySelector("#videoLabel").textContent = view.label;
  document.querySelectorAll(".view-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === key);
  });
  if (actions[key]) {
    commandInput.value = actions[key].prompt;
    timelineInput.value = actions[key].timeline;
    setState(...actions[key].state);
    updateCount();
  }
  if (key === "slideRightRaw") {
    setState(
      "RAW OSCAR / SLIDE RIGHT / USER REJECTED",
      "动作方向可测，但后半段机械手从五指整体退化成碎片状夹爪；该结果仅作为修复前证据。",
      "error",
    );
  }
  if (key === "repairComparison") {
    setState(
      "HISTORICAL REPAIRS / BOTH USER REJECTED",
      "左侧原始候选后半段手指碎裂；右侧固定拓扑候选虽保持轮廓，却让整只手像贴片一样刚性移位。两者仅作为失败证据。",
      "error",
    );
  }
  if (autoplay) mainVideo.play().catch(() => {});
}

function matchCompletedAction(text) {
  const compact = text.replace(/[\s，。；、,.!?！？]/g, "").toLowerCase();
  if (/(向左|移动到画面左|移到左侧|滑到左侧)/.test(compact)) return "slideLeft";
  if (/(向右|移动到画面右|移到右侧|滑到右侧)/.test(compact)) return "slideRight";
  if (/(抬|提起|上方|悬空|离开桌面)/.test(compact)) return "liftUp";
  return null;
}

function submitCommand(event) {
  event.preventDefault();
  const instruction = commandInput.value.trim();
  if (instruction.length < 8) {
    setState("ACTION CONTRACT / INVALID", "请明确动作对象、方向和结束状态。", "error");
    return;
  }
  const completedKey = matchCompletedAction(instruction);
  if (completedKey) {
    loadView(completedKey);
    return;
  }
  setState(
    "CUSTOM ACTION / NOT COMPILED",
    "该指令没有匹配到已实跑的动作契约。需要先编译新 control video，再启动模型生成和验收；页面不会返回伪造结果。",
    "error",
  );
}

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", () => loadView(button.dataset.action));
});
document.querySelectorAll(".view-button").forEach((button) => {
  button.addEventListener("click", () => loadView(button.dataset.view));
});
document.querySelectorAll(".action-card video, .case-card video, .archive-card video").forEach((video) => {
  video.addEventListener("mouseenter", () => video.play().catch(() => {}));
  video.addEventListener("mouseleave", () => {
    video.pause();
    video.currentTime = 0;
  });
});
commandInput.addEventListener("input", updateCount);
form.addEventListener("submit", submitCommand);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") form.requestSubmit();
});

fetch("showcase/oscar-acwm-evaluation.json?v=9")
  .then((response) => response.ok ? response.json() : Promise.reject(new Error("manifest unavailable")))
  .then((manifest) => {
    const variants = Object.fromEntries(manifest.variants.map((item) => [item.case_id, item]));
    const left = variants["slide-left"].scores.action_adherence;
    const right = manifest.articulated_carry.scores.action_adherence;
    const lift = variants["lift-up"].scores.action_adherence;
    document.querySelector("#evidenceStatus").textContent = manifest.status;
    document.querySelector("#rawAcceptanceValue").textContent = `${manifest.acceptance.raw_model_accepted_cases.length} / 3`;
    document.querySelector("#acceptanceValue").textContent = `${manifest.acceptance.workflow_accepted_cases.length} / 3`;
    document.querySelector("#alignmentValue").textContent = `${manifest.matched_protocol.frames} @ ${manifest.matched_protocol.fps} FPS`;
    document.querySelector("#rightMetric").textContent = `${right.toFixed(2)} · PASS`;
    document.querySelector("#structureMetric").textContent = `${manifest.articulated_carry.metrics.projected_progress_ratio.toFixed(2)}× · PASS`;
    document.querySelector("#liftMetric").textContent = `${lift.toFixed(2)} · PASS`;
    document.querySelector("#leftMetric").textContent = `${left.toFixed(2)} · FAIL`;
    document.querySelector("#rightActionScore").textContent = right.toFixed(2);
    document.querySelector("#liftActionScore").textContent = lift.toFixed(2);
    document.querySelector("#leftActionScore").textContent = left.toFixed(2);
  })
  .catch(() => {
    document.querySelector("#evidenceStatus").textContent = "PENDING SYNC";
  });

fetch("showcase/flower-task-vace-real-window-evaluation.json?v=9")
  .then((response) => response.ok ? response.json() : Promise.reject(new Error("training evidence unavailable")))
  .then((evidence) => {
    const semanticPasses = Object.values(evidence.semantic_gates || {}).filter(Boolean).length;
    const semanticTotal = Object.keys(evidence.semantic_gates || {}).length;
    const motionDelta = evidence.metrics?.adapted_vs_zero?.control_motion_alignment_delta;
    document.querySelector("#trainingDecision").textContent = evidence.decision.replaceAll("_", " ");
    document.querySelector("#trainingMotionDelta").textContent = Number.isFinite(motionDelta) ? motionDelta.toFixed(4) : "N/A";
    document.querySelector("#trainingSemanticGates").textContent = `${semanticPasses} / ${semanticTotal}`;
  })
  .catch(() => {
    document.querySelector("#trainingDecision").textContent = "EVIDENCE PENDING";
  });

updateCount();

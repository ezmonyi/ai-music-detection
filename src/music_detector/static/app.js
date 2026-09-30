"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = { artifacts: [], selected: new Set(), file: null, audioUrl: null, busy: false, loaded: false, task: null, result: null, pollTimer: null, clockTimer: null, startedAt: 0, failures: 0, maxBytes: null };
  const audioExtensions = /\.(wav|mp3|flac|ogg)$/i;
  const stageNames = { queued: "任务已排队，等待分析", uploading: "正在上传音频", validating: "检查音频与输入条件", decoding: "正在解码音频", preprocessing: "正在准备分析片段", preparing: "正在准备分析环境", downloading_models: "正在准备所需分析模型", separating: "正在分离音轨", separation: "正在分离音轨", extracting: "正在提取声学特征", features: "正在提取声学特征", scoring: "正在计算模型分数", calibrating: "正在计算校准结果", complete: "分析完成" };
  const statusNames = { ok: "已测量", eligible: "已测量", partial: "部分可测", available: "可用", complete: "已完成", success: "已完成", measured: "已测量", unavailable: "不可用", unsupported: "不支持", missing: "缺失", insufficient: "条件不足", insufficient_data: "条件不足", failed: "失败", error: "失败", skipped: "未执行", research_only: "研究性" };
  const element = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; };
  const printable = (value) => typeof value === "string" ? value : value === undefined || value === null ? "未提供" : JSON.stringify(value, null, 2);
  const finiteNumber = (value) => typeof value === "number" && Number.isFinite(value);
  const numberText = (value) => finiteNumber(value) ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 }).format(value) : "未提供";
  const bytesText = (bytes) => bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
  function showError(message) { $("form-error").textContent = message; $("form-error").hidden = !message; }
  function setResultStatus(text, kind = "neutral") { $("result-status").textContent = text; $("result-status").className = `badge ${kind}`; }

  async function api(url, options = {}, timeout = 30000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, headers: { Accept: "application/json", ...(options.headers || {}) } });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = body?.detail ?? body?.error ?? body?.message;
        throw new Error(detail ? printable(detail) : `服务请求失败（HTTP ${response.status}）`);
      }
      if (!body || typeof body !== "object") throw new Error("服务没有返回有效的分析数据。");
      return body;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("服务响应超时，请检查分析服务连接。");
      if (error instanceof TypeError) throw new Error("无法连接分析服务，请检查服务是否运行。");
      throw error;
    } finally { clearTimeout(timer); }
  }

  function eligible(artifact) { return Boolean(artifact.available) && (!artifact.research_only || $("research-opt-in").checked); }
  function updateControls() {
    $("selection-count").textContent = state.loaded ? `已选 ${state.selected.size} 项` : "正在读取可用特征…";
    $("submit-button").disabled = state.busy || !state.loaded || !state.file || state.selected.size === 0;
    $("submit-button").querySelector("span").textContent = state.busy ? "正在分析…" : "开始分析";
    $("audio-input").disabled = state.busy;
    $("remove-file").disabled = state.busy;
    $("research-opt-in").disabled = state.busy;
    $("select-all").disabled = state.busy || !state.loaded;
    $("clear-selection").disabled = state.busy || state.selected.size === 0;
    $("reload-artifacts").disabled = state.busy;
    $("drop-zone").classList.toggle("disabled", state.busy);
    $("analysis-form").setAttribute("aria-busy", String(state.busy));
    $("artifact-options").querySelectorAll("input").forEach((input) => {
      const artifact = state.artifacts.find((a) => a.id === input.value);
      input.disabled = state.busy || !artifact || !eligible(artifact);
      input.checked = state.selected.has(input.value);
    });
    if (!state.busy && state.result) {
      const previous = [...(state.result.selected_artifacts || [])].sort().join(',');
      const current = [...state.selected].sort().join(',');
      setResultStatus(previous === current ? "分析完成" : "选择已更改 · 下方仍为上次结果", previous === current ? "success" : "warning");
    }
  }

  function renderArtifactOptions() {
    const root = $("artifact-options"); root.replaceChildren();
    for (const artifact of state.artifacts) {
      const label = element("label", "artifact-option");
      const input = element("input"); input.type = "checkbox"; input.value = artifact.id; input.name = "artifact";
      input.addEventListener("change", () => { input.checked ? state.selected.add(artifact.id) : state.selected.delete(artifact.id); showError(""); updateControls(); });
      const info = element("span"); const heading = element("span", "artifact-option-head");
      heading.append(element("span", "artifact-id", artifact.id), element("span", "artifact-name", artifact.name || artifact.id));
      if (artifact.research_only) heading.append(element("span", "artifact-tag", "研究性"));
      info.append(heading);
      if (artifact.description) info.append(element("span", "artifact-description", artifact.description));
      if (!artifact.available) info.append(element("span", "artifact-reason", artifact.reason || "当前环境不可用"));
      else if (artifact.research_only && !$("research-opt-in").checked) info.append(element("span", "artifact-reason", "需先开启研究性特征"));
      label.append(input, info); root.append(label);
    }
    if (!state.artifacts.length) root.append(element("p", "muted loading-copy", "服务目前未提供可用特征。"));
    $("research-control").hidden = !state.artifacts.some((a) => a.research_only);
    updateControls();
  }

  async function loadArtifacts() {
    if (state.busy) return;
    const wasLoaded = state.loaded; state.loaded = false; updateControls();
    $("connection-label").textContent = "正在检查分析服务"; $("connection-dot").className = "status-dot";
    try {
      const data = await api("/api/artifacts");
      if (!Array.isArray(data.artifacts)) throw new Error("特征目录格式不正确，请检查服务配置。");
      state.artifacts = data.artifacts.filter((a) => a && typeof a.id === "string");
      state.loaded = true;
      state.selected = new Set(state.artifacts.filter((a) => eligible(a) && (wasLoaded ? state.selected.has(a.id) : !a.research_only)).map((a) => a.id));
      state.maxBytes = data.model_status?.upload_max_bytes ?? data.model_status?.max_upload_bytes ?? null;
      $("model-status").textContent = printable(data.model_status);
      $("connection-label").textContent = "分析服务已连接"; $("connection-dot").className = "status-dot ready";
      $("upload-help").textContent = `WAV、MP3、FLAC、OGG · 原生双声道 · 至少 30 秒${finiteNumber(state.maxBytes) ? ` · 上限 ${bytesText(state.maxBytes)}` : ""}`;
      showError(""); renderArtifactOptions();
      if (state.file && finiteNumber(state.maxBytes) && state.file.size > state.maxBytes) showError(`所选文件超过服务配置的 ${bytesText(state.maxBytes)} 限制，请选择其他文件。`);
    } catch (error) {
      $("connection-label").textContent = "分析服务连接失败"; $("connection-dot").className = "status-dot error";
      $("artifact-options").replaceChildren(element("p", "muted loading-copy", "暂时无法读取特征目录。请重新检查服务连接。"));
      $("model-status").textContent = error.message; showError(error.message); updateControls();
    }
  }

  function chooseFile(file) {
    if (state.busy || !file) return;
    if (!file.size) { showError("文件为空，请选择有效的音频文件。"); return; }
    if (!audioExtensions.test(file.name)) { showError("请选择 WAV、MP3、FLAC 或 OGG 文件；服务端会再次检查实际格式。"); return; }
    if (finiteNumber(state.maxBytes) && file.size > state.maxBytes) { showError(`文件超过服务配置的 ${bytesText(state.maxBytes)} 限制。`); return; }
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.file = file; state.audioUrl = URL.createObjectURL(file);
    state.result = null; state.task = null;
    $("results").hidden = true; $("progress-panel").hidden = true; $("empty-state").hidden = false;
    $("file-name").textContent = file.name; $("file-size").textContent = bytesText(file.size); $("file-card").hidden = false;
    $("audio-preview").src = state.audioUrl; $("audio-preview").hidden = false;
    showError(""); updateControls();
    setResultStatus("音频已选择");
  }

  function startClock() {
    clearInterval(state.clockTimer); state.startedAt = Date.now();
    const update = () => { const seconds = Math.floor((Date.now() - state.startedAt) / 1000); $("elapsed-time").textContent = seconds < 60 ? `已用时 ${seconds} 秒` : `已用时 ${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒`; };
    update(); state.clockTimer = setInterval(update, 1000);
  }
  function stopClock() { clearInterval(state.clockTimer); state.clockTimer = null; }
  function stopPolling() { clearTimeout(state.pollTimer); state.pollTimer = null; }

  async function submit(event) {
    event.preventDefault(); if (state.busy) return;
    if (!state.file) { showError("请先选择一份音频。"); return; }
    const selected = state.artifacts.filter((a) => state.selected.has(a.id) && eligible(a)).map((a) => a.id);
    if (!selected.length) { showError("请选择至少一项当前可用的声学特征。"); return; }
    if (finiteNumber(state.maxBytes) && state.file.size > state.maxBytes) { showError(`文件超过服务配置的 ${bytesText(state.maxBytes)} 限制。`); return; }
    state.busy = true; state.result = null; state.task = null; state.failures = 0; stopPolling(); showError(""); updateControls();
    $("empty-state").hidden = true; $("results").hidden = true; $("progress-panel").hidden = false;
    $("progress-panel").classList.remove("paused"); $("progress-error").hidden = true; $("retry-poll").hidden = true;
    $("progress-title").textContent = "正在提交音频"; $("progress-stage").textContent = "正在上传文件并创建分析任务…"; $("task-id").textContent = "";
    setResultStatus("正在上传", "processing"); startClock();
    const payload = new FormData(); payload.append("audio", state.file); payload.append("artifacts", JSON.stringify(selected)); payload.append("research", String($("research-opt-in").checked));
    try {
      const task = await api("/api/analyses", { method: "POST", body: payload }, 180000);
      if (typeof task.id !== "string" || !task.id) throw new Error("服务未返回有效任务编号。");
      state.task = task.id; $("task-id").textContent = `任务 ${task.id}`;
      await poll();
    } catch (error) { terminalFailure(error.message); }
  }

  function terminalFailure(message) {
    state.busy = false; stopPolling(); stopClock(); updateControls();
    $("progress-panel").classList.add("paused"); $("progress-title").textContent = "本次分析未完成";
    $("progress-stage").textContent = "修正问题后，可以重新提交分析。";
    $("progress-error").textContent = message; $("progress-error").hidden = false;
    $("retry-poll").hidden = true; setResultStatus("分析未完成", "error");
  }

  async function poll() {
    if (!state.task) return;
    const taskId = state.task;
    try {
      const task = await api(`/api/analyses/${encodeURIComponent(taskId)}`);
      if (taskId !== state.task) return;
      state.failures = 0; $("progress-error").hidden = true; $("retry-poll").hidden = true; $("progress-panel").classList.remove("paused");
      if (task.status === "complete") {
        if (!task.result || typeof task.result !== "object") throw new Error("任务已完成，但服务没有返回结果内容。");
        state.result = task.result; state.busy = false; stopPolling(); stopClock(); updateControls(); renderResult(task.result); return;
      }
      if (task.status === "failed") { terminalFailure(printable(task.error || "分析服务报告任务失败。")); return; }
      if (!["queued", "running"].includes(task.status)) throw new Error(`未知任务状态：${printable(task.status)}`);
      $("progress-title").textContent = task.status === "queued" ? "任务正在排队" : "正在分析音频";
      $("progress-stage").textContent = stageNames[task.stage] || printable(task.stage || stageNames[task.status] || "服务正在处理所选特征");
      setResultStatus(task.status === "queued" ? "排队中" : "分析中", "processing");
      state.pollTimer = setTimeout(poll, task.status === "queued" ? 2200 : 1300);
    } catch (error) {
      if (taskId !== state.task) return;
      state.failures += 1;
      if (state.failures < 4) { $("progress-stage").textContent = "连接暂时中断，正在重新获取任务状态…"; state.pollTimer = setTimeout(poll, 2500 * state.failures); }
      else {
        stopPolling(); $("progress-panel").classList.add("paused");
        $("progress-title").textContent = "暂时无法读取任务状态";
        $("progress-stage").textContent = "服务端任务可能仍在运行；重新连接不会重复提交音频。";
        $("progress-error").textContent = error.message; $("progress-error").hidden = false; $("retry-poll").hidden = false;
        setResultStatus("连接中断", "warning");
      }
    }
  }

  function decisionText(value) {
    const names = { ai: "模型判断：倾向 AI 标签", likely_ai: "模型判断：倾向 AI 标签", AI: "模型判断：倾向 AI 标签", human: "模型判断：倾向人类标签", likely_human: "模型判断：倾向人类标签", Human: "模型判断：倾向人类标签", uncertain: "模型判断：不确定", abstain: "模型暂不作出判断", unavailable: "当前无法给出判断" };
    return names[value] || (value ? printable(value) : "服务未提供分类判断");
  }
  function calibrationText(value) {
    if (value === "abstained_no_observed_predictive_features") return "无有效指标 · 不作判定";
    if (value === "fitted_dev_calibration_evaluated_untouched_internal_dev_test") return "开发集独立分区校准 · 仅内部测试";
    const names = { calibrated: "已应用概率校准", available: "已提供校准结果", uncalibrated: "尚未进行概率校准", unavailable: "校准结果不可用", not_calibrated: "尚未进行概率校准", not_available: "校准结果不可用" };
    return names[value] || (value ? printable(value) : "未提供校准状态");
  }
  function renderResult(result) {
    $("progress-panel").hidden = true; $("results").hidden = false; setResultStatus("分析完成", "success");
    const prediction = result.prediction || {}; const probability = prediction.probability;
    const hasProbability = finiteNumber(probability) && probability >= 0 && probability <= 1;
    $("prediction-title").textContent = hasProbability ? "AI 标签的校准概率" : "AI 标签概率";
    $("probability-value").textContent = hasProbability ? `${(probability * 100).toFixed(1)}%` : "暂不可用";
    $("probability-value").classList.toggle("unavailable", !hasProbability);
    $("prediction-decision").textContent = decisionText(prediction.decision);
    $("calibration-label").textContent = calibrationText(prediction.calibration_status);
    $("probability-meter").hidden = !hasProbability; $("probability-fill").style.width = hasProbability ? `${probability * 100}%` : "0%";
    $("prediction-scope").textContent = prediction.scope === "new_retrospective_internal_development_experiment_not_external_validation" ? "概率仅对应类别、来源及依赖组平衡后的研究参考分布；不是日常歌曲总体的真实 AI 占比，也不是未知生成模型上的外部验证。判定使用校准概率 0.5 阈值。" : prediction.scope ? printable(prediction.scope) : "服务未提供此结果的适用范围。请结合下方证据与处理记录解读。";
    $("raw-score").textContent = numberText(prediction.raw_score); $("score-threshold").textContent = numberText(prediction.decision_threshold);
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    $("warnings-list").replaceChildren(...warnings.map((warning) => element("li", "", printable(warning)))); $("warnings-panel").hidden = !warnings.length;
    const audio = result.audio || {}; $("result-audio-name").textContent = audio.name || state.file?.name || "未提供文件名";
    const facts = [["时长", finiteNumber(audio.duration) ? `${numberText(audio.duration)} 秒` : "未提供"], ["采样率", finiteNumber(audio.sample_rate) ? `${numberText(audio.sample_rate)} Hz` : "未提供"], ["声道", finiteNumber(audio.channels) ? String(audio.channels) : "未提供"]];
    $("audio-facts").replaceChildren(...facts.map(([name, value]) => { const div = element("div"); div.append(element("dt", "", name), element("dd", "", value)); return div; }));
    $("audio-sha").textContent = audio.sha256 || "服务未提供文件哈希";
    const features = Array.isArray(result.features) ? result.features : [];
    $("evidence-count").textContent = `${features.length} 项返回记录`;
    $("feature-cards").replaceChildren(...features.map(renderFeature));
    if (!features.length) $("feature-cards").append(element("p", "muted", "服务未返回逐项测量记录。"));
    $("provenance-json").textContent = JSON.stringify({ analysis_id: state.task, selected_artifacts: result.selected_artifacts, provenance: result.provenance || {} }, null, 2);
  }

  function renderFeature(feature) {
    const card = element("article", "feature-card"); const head = element("div", "feature-card-head"); const name = element("div");
    name.append(element("span", "feature-card-id", feature.id || ""), element("h4", "", feature.name || feature.id || "声学特征"));
    const good = ["ok", "available", "complete", "success", "measured"].includes(feature.status);
    const bad = ["failed", "error"].includes(feature.status);
    head.append(name, element("span", `badge ${good ? "success" : bad ? "error" : "warning"}`, statusNames[feature.status] || printable(feature.status)));
    card.append(head);
    if (Array.isArray(feature.metrics) && feature.metrics.length) {
      const metrics = element("dl", "metrics");
      for (const metric of feature.metrics) {
        const row = element("div", "metric");
        const value = finiteNumber(metric.value) ? numberText(metric.value) : metric.value === null || metric.value === undefined ? "未观测" : printable(metric.value);
        row.append(element("dt", "", metric.name || "指标"), element("dd", "", `${value}${metric.unit && metric.value !== null && metric.value !== undefined ? ` ${metric.unit}` : ""}`)); metrics.append(row);
      }
      card.append(metrics);
    }
    if (feature.reason) card.append(element("p", "feature-reason", printable(feature.reason)));
    if (feature.contribution !== undefined && feature.contribution !== null) {
      const detail = element("details", "contribution"); detail.append(element("summary", "", "模型中的特征贡献与测量证据"), element("pre", "", JSON.stringify({total: feature.contribution, value_terms: feature.value_contribution, missingness_terms: feature.missingness_contribution, strongest_measurements: feature.evidence}, null, 2)), element("p", "", "贡献描述模型内部的原始分数构成，不是概率百分点，也不表示声学现象与创作方式之间的因果关系。")); card.append(detail);
    }
    return card;
  }

  $("analysis-form").addEventListener("submit", submit);
  $("audio-input").addEventListener("change", (event) => { chooseFile(event.target.files?.[0]); event.target.value = ""; });
  $("remove-file").addEventListener("click", () => {
    if (state.busy) return; $("audio-preview").pause(); $("audio-preview").removeAttribute("src"); $("audio-preview").load();
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl); state.audioUrl = null; state.file = null;
    $("file-card").hidden = true; $("audio-preview").hidden = true; showError(""); updateControls(); if (!state.result) setResultStatus("等待音频");
  });
  for (const eventName of ["dragenter", "dragover"]) $("drop-zone").addEventListener(eventName, (event) => { event.preventDefault(); if (!state.busy) $("drop-zone").classList.add("drag-over"); });
  for (const eventName of ["dragleave", "drop"]) $("drop-zone").addEventListener(eventName, (event) => { event.preventDefault(); $("drop-zone").classList.remove("drag-over"); });
  $("drop-zone").addEventListener("drop", (event) => { if (event.dataTransfer.files.length > 1) { showError("每次分析一份音频，请只拖入一个文件。"); return; } chooseFile(event.dataTransfer.files[0]); });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  $("select-all").addEventListener("click", () => { state.selected = new Set(state.artifacts.filter(eligible).map((a) => a.id)); updateControls(); });
  $("clear-selection").addEventListener("click", () => { state.selected.clear(); updateControls(); });
  $("research-opt-in").addEventListener("change", () => { for (const artifact of state.artifacts) if (!eligible(artifact)) state.selected.delete(artifact.id); renderArtifactOptions(); });
  $("reload-artifacts").addEventListener("click", loadArtifacts);
  $("retry-poll").addEventListener("click", () => { state.failures = 0; $("retry-poll").hidden = true; $("progress-error").hidden = true; $("progress-panel").classList.remove("paused"); poll(); });
  $("download-result").addEventListener("click", () => {
    if (!state.result) return;
    const link = element("a"); link.href = `/api/analyses/${encodeURIComponent(state.task)}/download`;
    link.download = `music-evidence-${String(state.task).replace(/[^a-zA-Z0-9_-]/g, "_")}.json`;
    document.body.append(link); link.click(); link.remove();
  });
  window.addEventListener("beforeunload", () => { stopPolling(); stopClock(); if (state.audioUrl) URL.revokeObjectURL(state.audioUrl); });
  loadArtifacts();
})();

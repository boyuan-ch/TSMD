"use strict";

const MANIFEST_URL = "data/manifest.json";
const BASE_MARGIN = { top: 18, right: 18, bottom: 36, left: 48 };
const DROP_ROW_HEIGHT = 17;

const state = {
  manifest: null,
  activeDemo: null,
  activeVideo: null,
  data: null,
  player: null,
  playerState: -1,
  mediaReady: false,
  pendingSeek: 0,
  showClean: true,
  draggingSlider: false,
  draggingChart: false,
  resizingChart: false,
  chartResizeStartY: 0,
  chartResizeStartHeight: 0,
  scrubbing: null,
  resumeAfterScrub: false,
  animationFrame: 0,
  lastAnimationRender: 0,
  resizeFrame: 0,
  canvasWidth: 0,
  canvasHeight: 0,
  canvasRatio: 0,
  requestToken: 0,
};

const elements = {
  demoTabs: document.querySelector("#demo-tabs"),
  videoPicker: document.querySelector("#video-picker"),
  cleanToggleWrap: document.querySelector("#clean-toggle-wrap"),
  cleanToggle: document.querySelector("#clean-toggle"),
  pageEyebrow: document.querySelector("#page-eyebrow"),
  demoKicker: document.querySelector("#demo-kicker"),
  demoTitle: document.querySelector("#demo-title"),
  demoDescription: document.querySelector("#demo-description"),
  metricStrip: document.querySelector("#metric-strip"),
  meters: document.querySelector("#meters"),
  chartLegend: document.querySelector("#chart-legend"),
  playerMount: document.querySelector("#youtube-player"),
  consentPanel: document.querySelector("#consent-panel"),
  consentButton: document.querySelector("#consent-button"),
  videoSourceLink: document.querySelector("#video-source-link"),
  footerVideoLink: document.querySelector("#footer-video-link"),
  playButton: document.querySelector("#play-button"),
  seekSlider: document.querySelector("#seek-slider"),
  speedSelect: document.querySelector("#speed-select"),
  currentTime: document.querySelector("#current-time"),
  durationTime: document.querySelector("#duration-time"),
  headerTime: document.querySelector("#header-time"),
  dropStatus: document.querySelector("#drop-status"),
  dropLabel: document.querySelector("#drop-label"),
  chartWrap: document.querySelector("#chart-wrap"),
  chartResizeHandle: document.querySelector("#chart-resize-handle"),
  canvas: document.querySelector("#salience-chart"),
  tooltip: document.querySelector("#chart-tooltip"),
  errorBanner: document.querySelector("#error-banner"),
};

const context = elements.canvas.getContext("2d");

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function hideError() {
  elements.errorBanner.hidden = true;
}

function formatTime(value) {
  if (!Number.isFinite(value)) return "--:--";
  const seconds = Math.max(0, Math.floor(value));
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function dropKeySet() {
  return new Set((state.data?.dropRows || []).map((row) => row.key));
}

function allSeriesKeys() {
  if (!state.data) return [];
  const dropKeys = dropKeySet();
  return Object.keys(state.data.series).filter((key) => !dropKeys.has(key));
}

function visibleSeriesKeys() {
  return allSeriesKeys().filter((key) => state.showClean || !state.data.series[key].clean);
}

function timelineEnd() {
  return state.data ? state.data.samples.at(-1).second : 0;
}

function playbackEnd() {
  if (!state.mediaReady) return timelineEnd();
  const duration = Number(state.player.getDuration());
  return Number.isFinite(duration) && duration > 0
    ? Math.min(duration, timelineEnd())
    : timelineEnd();
}

function currentPlaybackTime() {
  if (!state.mediaReady) return state.pendingSeek || 0;
  return state.player.getCurrentTime() || 0;
}

function isPlaying() {
  return window.YT && state.playerState === YT.PlayerState.PLAYING;
}

function clampTime(value) {
  return Math.min(Math.max(Number(value) || 0, 0), playbackEnd());
}

function interpolatedSample(time) {
  const samples = state.data.samples;
  const numericTime = Number(time);
  const bounded = Math.min(
    Math.max(Number.isFinite(numericTime) ? numericTime : 0, 0),
    samples.length - 1,
  );
  const lowerIndex = Math.floor(bounded);
  const upperIndex = Math.min(lowerIndex + 1, samples.length - 1);
  const blend = bounded - lowerIndex;
  const lower = samples[lowerIndex];
  const upper = samples[upperIndex];
  const result = { second: bounded };
  for (const key of allSeriesKeys()) {
    result[key] = lower[key] + (upper[key] - lower[key]) * blend;
  }
  for (const row of state.data.dropRows || []) result[row.key] = lower[row.key];
  return result;
}

function meterId(key) {
  return key.replace(/[^a-z0-9_-]/gi, "-");
}

function rebuildMeters() {
  elements.meters.replaceChildren();
  for (const key of visibleSeriesKeys()) {
    const series = state.data.series[key];
    const article = document.createElement("article");
    article.className = `meter${series.clean ? " is-clean" : ""}`;
    article.dataset.series = key;
    article.innerHTML = `
      <div class="meter-copy">
        <span class="series-name"><span class="series-swatch"></span>${series.label}</span>
        <strong class="meter-value" id="value-${meterId(key)}">0.000</strong>
      </div>
      <div class="meter-track" role="meter" aria-label="${series.label} salience" aria-valuemin="0" aria-valuemax="1" aria-valuenow="0">
        <span class="meter-fill"></span><span class="meter-marker"></span>
      </div>`;
    for (const node of article.querySelectorAll(".series-swatch, .meter-fill, .meter-marker")) {
      node.style.background = series.color;
    }
    elements.meters.append(article);
  }
}

function rebuildLegend() {
  elements.chartLegend.replaceChildren();
  for (const key of visibleSeriesKeys()) {
    const series = state.data.series[key];
    const item = document.createElement("span");
    const line = document.createElement("i");
    line.className = `legend-line${series.clean ? " is-dashed" : ""}`;
    line.style.setProperty("--series-color", series.color);
    item.append(line, series.label);
    elements.chartLegend.append(item);
  }
  for (const row of state.data.dropRows || []) {
    const item = document.createElement("span");
    const block = document.createElement("i");
    block.className = "legend-drop";
    block.style.setProperty("--drop-color", row.color);
    item.append(block, `${row.name} drop`);
    elements.chartLegend.append(item);
  }
}

function updateMeters(sample) {
  for (const key of visibleSeriesKeys()) {
    const meter = elements.meters.querySelector(`[data-series="${key}"]`);
    if (!meter) continue;
    const value = Math.min(Math.max(sample[key], 0), 1);
    const percent = `${(value * 100).toFixed(2)}%`;
    meter.querySelector(".meter-value").textContent = value.toFixed(3);
    meter.querySelector(".meter-fill").style.width = percent;
    meter.querySelector(".meter-marker").style.left = percent;
    meter.querySelector(".meter-track").setAttribute("aria-valuenow", value.toFixed(3));
  }
}

function updateDropState(sample) {
  const activeRows = (state.data.dropRows || []).filter((row) => Boolean(sample[row.key]));
  elements.dropStatus.dataset.dropped = String(activeRows.length > 0);
  elements.dropLabel.textContent = activeRows.length
    ? `${activeRows.map((row) => row.label).join(" + ")} dropped`
    : "All modalities retained";
}

function updateTimes(time) {
  const duration = playbackEnd();
  elements.currentTime.textContent = formatTime(time);
  elements.durationTime.textContent = formatTime(duration);
  elements.headerTime.textContent = `${formatTime(time)} / ${formatTime(duration)}`;
  if (!state.draggingSlider) elements.seekSlider.value = String(time);
}

function render(time = currentPlaybackTime()) {
  if (!state.data) return;
  const bounded = clampTime(time);
  const sample = interpolatedSample(bounded);
  updateMeters(sample);
  updateDropState(sample);
  updateTimes(bounded);
  drawChart(bounded);
}

function chartLayout(width, height) {
  const stripHeight = (state.data?.dropRows?.length || 0) * DROP_ROW_HEIGHT;
  const margin = { ...BASE_MARGIN, bottom: BASE_MARGIN.bottom + stripHeight };
  return {
    margin,
    plotRight: width - margin.right,
    plotBottom: height - margin.bottom,
    stripTop: height - margin.bottom + 27,
  };
}

function xForTime(time, width, layout) {
  const plotWidth = width - layout.margin.left - layout.margin.right;
  return layout.margin.left + (time / Math.max(timelineEnd(), 1)) * plotWidth;
}

function yForValue(value, layout) {
  const plotHeight = layout.plotBottom - layout.margin.top;
  return layout.margin.top + (1 - value) * plotHeight;
}

function setupCanvas() {
  const rect = elements.chartWrap.getBoundingClientRect();
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (width !== state.canvasWidth || height !== state.canvasHeight || ratio !== state.canvasRatio) {
    elements.canvas.width = Math.round(width * ratio);
    elements.canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    state.canvasWidth = width;
    state.canvasHeight = height;
    state.canvasRatio = ratio;
  }
  return { width, height };
}

function drawAxes(width, layout) {
  context.save();
  context.font = '11px "IBM Plex Mono", "Cascadia Mono", monospace';
  context.lineWidth = 1;
  context.textBaseline = "middle";
  for (let value = 0; value <= 1.0001; value += 0.2) {
    const y = yForValue(value, layout);
    context.strokeStyle = value === 0 ? "#b9bbb8" : "#e4e4e0";
    context.beginPath();
    context.moveTo(layout.margin.left, y + 0.5);
    context.lineTo(layout.plotRight, y + 0.5);
    context.stroke();
    context.fillStyle = "#6a6f73";
    context.textAlign = "right";
    context.fillText(value.toFixed(1), layout.margin.left - 8, y);
  }
  const tickStep = width < 700 ? 60 : 30;
  context.textBaseline = "top";
  context.textAlign = "center";
  for (let second = 0; second <= timelineEnd(); second += tickStep) {
    const x = xForTime(second, width, layout);
    context.strokeStyle = "#d4d5d1";
    context.beginPath();
    context.moveTo(x + 0.5, layout.plotBottom);
    context.lineTo(x + 0.5, layout.plotBottom + 5);
    context.stroke();
    context.fillStyle = "#6a6f73";
    context.textAlign = second === 0 ? "left" : "center";
    context.fillText(formatTime(second), x, layout.plotBottom + 9);
  }
  context.restore();
}

function drawDropRows(width, layout) {
  const rows = state.data.dropRows || [];
  context.save();
  context.font = '600 10px "IBM Plex Mono", "Cascadia Mono", monospace';
  context.textAlign = "right";
  context.textBaseline = "middle";
  rows.forEach((row, rowIndex) => {
    const y = layout.stripTop + rowIndex * DROP_ROW_HEIGHT;
    context.fillStyle = "#f0f0ed";
    context.fillRect(layout.margin.left, y, layout.plotRight - layout.margin.left, 11);
    context.fillStyle = row.color;
    context.fillText(row.label, layout.margin.left - 8, y + 5.5);
    context.globalAlpha = 0.9;
    for (const sample of state.data.samples) {
      if (!sample[row.key]) continue;
      const left = xForTime(Math.max(0, sample.second - 0.5), width, layout);
      const right = xForTime(Math.min(timelineEnd(), sample.second + 0.5), width, layout);
      context.fillRect(left, y, Math.max(1, right - left), 11);
    }
    context.globalAlpha = 1;
  });
  context.restore();
}

function drawCurve(key, currentTime, width, layout) {
  const samples = state.data.samples;
  const series = state.data.series[key];
  const lastFullIndex = Math.min(Math.floor(currentTime), samples.length - 1);
  context.save();
  context.beginPath();
  context.rect(
    layout.margin.left,
    layout.margin.top,
    layout.plotRight - layout.margin.left,
    layout.plotBottom - layout.margin.top,
  );
  context.clip();
  context.beginPath();
  for (let index = 0; index <= lastFullIndex; index += 1) {
    const sample = samples[index];
    const x = xForTime(sample.second, width, layout);
    const y = yForValue(sample[key], layout);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  if (currentTime > lastFullIndex && lastFullIndex < samples.length - 1) {
    const sample = interpolatedSample(currentTime);
    context.lineTo(xForTime(currentTime, width, layout), yForValue(sample[key], layout));
  }
  context.strokeStyle = series.color;
  context.lineWidth = key === "gt" ? 2.5 : 2.1;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.setLineDash(series.clean || series.dash === "dashed" ? [8, 5] : []);
  context.stroke();
  context.restore();
}

function drawCursor(currentTime, width, layout) {
  const x = xForTime(currentTime, width, layout);
  const sample = interpolatedSample(currentTime);
  context.save();
  context.strokeStyle = "rgba(32, 33, 36, 0.65)";
  context.setLineDash([4, 4]);
  context.beginPath();
  context.moveTo(x + 0.5, layout.margin.top);
  context.lineTo(x + 0.5, layout.plotBottom);
  context.stroke();
  context.setLineDash([]);
  for (const key of visibleSeriesKeys()) {
    context.beginPath();
    context.arc(x, yForValue(sample[key], layout), 4, 0, Math.PI * 2);
    context.fillStyle = state.data.series[key].color;
    context.fill();
    context.strokeStyle = "#fff";
    context.lineWidth = 1.5;
    context.stroke();
  }
  context.restore();
}

function drawChart(time) {
  if (!state.data) return;
  const { width, height } = setupCanvas();
  const layout = chartLayout(width, height);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fff";
  context.fillRect(0, 0, width, height);
  drawAxes(width, layout);
  for (const key of visibleSeriesKeys()) drawCurve(key, time, width, layout);
  drawCursor(time, width, layout);
  drawDropRows(width, layout);
}

function animationLoop(timestamp) {
  if (timestamp - state.lastAnimationRender >= 1000 / 30) {
    render();
    state.lastAnimationRender = timestamp;
  }
  if (isPlaying()) state.animationFrame = requestAnimationFrame(animationLoop);
}

function setPlayingState() {
  const playing = isPlaying();
  elements.playButton.classList.toggle("is-playing", playing);
  elements.playButton.setAttribute("aria-label", playing ? "Pause video" : "Play video");
  cancelAnimationFrame(state.animationFrame);
  if (playing) state.animationFrame = requestAnimationFrame(animationLoop);
  else render();
}

function togglePlayback() {
  if (!state.mediaReady || state.scrubbing) return;
  if (isPlaying()) state.player.pauseVideo();
  else state.player.playVideo();
}

function beginScrub(source) {
  if (state.scrubbing) return;
  state.scrubbing = source;
  state.resumeAfterScrub = isPlaying();
  if (state.resumeAfterScrub) state.player.pauseVideo();
}

function endScrub(source) {
  if (state.scrubbing !== source) return;
  state.scrubbing = null;
  if (state.resumeAfterScrub && state.mediaReady) state.player.playVideo();
  state.resumeAfterScrub = false;
  render();
}

function seekTo(time) {
  const bounded = clampTime(time);
  state.pendingSeek = bounded;
  if (state.mediaReady) state.player.seekTo(bounded, true);
  render(bounded);
}

function chartTimeFromPointer(event) {
  const rect = elements.canvas.getBoundingClientRect();
  const layout = chartLayout(rect.width, rect.height);
  const x = Math.min(Math.max(event.clientX - rect.left, layout.margin.left), layout.plotRight);
  return ((x - layout.margin.left) / (layout.plotRight - layout.margin.left)) * timelineEnd();
}

function positionTooltip(event, time) {
  const sample = interpolatedSample(time);
  elements.tooltip.innerHTML = [
    `<strong>${formatTime(time)}</strong>`,
    ...visibleSeriesKeys().map((key) => `${state.data.series[key].label}&nbsp;&nbsp;${sample[key].toFixed(3)}`),
  ].join("<br>");
  elements.tooltip.hidden = false;
  const wrapRect = elements.chartWrap.getBoundingClientRect();
  const localX = event.clientX - wrapRect.left;
  const localY = event.clientY - wrapRect.top;
  elements.tooltip.style.left = `${Math.min(Math.max(localX + 12, 8), wrapRect.width - elements.tooltip.offsetWidth - 8)}px`;
  elements.tooltip.style.top = `${Math.min(Math.max(localY - 38, 8), wrapRect.height - elements.tooltip.offsetHeight - 8)}px`;
}

function onChartPointer(event, seek) {
  if (!state.data) return;
  const time = chartTimeFromPointer(event);
  positionTooltip(event, time);
  if (seek) seekTo(time);
}

function setChartHeight(height) {
  const limit = Math.max(260, Math.min(760, Math.round(window.innerHeight * 0.8)));
  elements.chartWrap.style.setProperty("--chart-height", `${Math.min(Math.max(Math.round(height), 260), limit)}px`);
  render();
}

function renderMetrics() {
  const metrics = state.data.metrics;
  elements.metricStrip.hidden = !metrics;
  elements.metricStrip.replaceChildren();
  if (!metrics) return;
  const items = [
    ["TripleSumm (Baseline) clean", metrics.triplesummClean.mAP15, true],
    ["TripleSumm (Baseline) drop", metrics.triplesummDrop.mAP15, false],
    ["TSMD-Mix (Ours) clean", metrics.tsmdMixClean.mAP15, true],
    ["TSMD-Mix (Ours) drop", metrics.tsmdMixDrop.mAP15, false],
  ];
  for (const [label, value, clean] of items) {
    if (clean && !state.showClean) continue;
    const item = document.createElement("span");
    item.innerHTML = `<small>${label}</small><strong>${value.toFixed(1)}</strong>`;
    elements.metricStrip.append(item);
  }
}

function renderNavigation() {
  elements.demoTabs.replaceChildren();
  elements.demoTabs.hidden = state.manifest.demos.length <= 1;
  for (const demo of state.manifest.demos) {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "tab";
    button.className = "tab-button";
    button.textContent = demo.label;
    button.dataset.active = String(demo.id === state.activeDemo.id);
    button.setAttribute("aria-selected", String(demo.id === state.activeDemo.id));
    button.addEventListener("click", () => selectDemo(demo.id));
    elements.demoTabs.append(button);
  }
  elements.videoPicker.replaceChildren();
  elements.videoPicker.hidden = state.activeDemo.videos.length <= 1;
  for (const video of state.activeDemo.videos) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "video-button";
    button.textContent = video.label;
    button.dataset.active = String(video.id === state.activeVideo.id);
    button.setAttribute("aria-pressed", String(video.id === state.activeVideo.id));
    button.addEventListener("click", () => selectVideo(video.id));
    elements.videoPicker.append(button);
  }
  elements.cleanToggleWrap.hidden = !allSeriesKeys().some((key) => state.data.series[key].clean);
}

function updatePlayerVideo() {
  const videoUrl = `https://www.youtube.com/watch?v=${encodeURIComponent(state.data.videoId)}`;
  elements.videoSourceLink.href = videoUrl;
  elements.footerVideoLink.href = videoUrl;
  if (state.player && state.mediaReady) {
    state.player.cueVideoById({ videoId: state.data.videoId, startSeconds: 0 });
  }
}

async function selectVideo(videoId) {
  const video = state.activeDemo.videos.find((item) => item.id === videoId);
  if (!video) return;
  const token = ++state.requestToken;
  hideError();
  const response = await fetch(video.path);
  if (!response.ok) throw new Error(`score request returned ${response.status}`);
  const data = await response.json();
  if (token !== state.requestToken) return;
  if (!Array.isArray(data.samples) || data.samples.length < 2) throw new Error("score data is empty or malformed");
  state.activeVideo = video;
  state.data = data;
  state.pendingSeek = 0;
  state.playerState = -1;
  elements.seekSlider.value = "0";
  elements.seekSlider.max = String(timelineEnd());
  elements.pageEyebrow.textContent = state.activeDemo.label;
  elements.demoKicker.textContent = "Robustness example";
  elements.demoTitle.textContent = video.label;
  elements.demoDescription.textContent = "";
  renderMetrics();
  rebuildMeters();
  rebuildLegend();
  renderNavigation();
  updatePlayerVideo();
  render(0);
}

async function selectDemo(demoId) {
  const demo = state.manifest.demos.find((item) => item.id === demoId);
  if (!demo || demo.id === state.activeDemo?.id) return;
  state.activeDemo = demo;
  state.activeVideo = null;
  try {
    await selectVideo(demo.videos[0].id);
  } catch (error) {
    showError(`Could not load demo data: ${error.message}`);
  }
}

function populatePlaybackRates() {
  const rates = state.player.getAvailablePlaybackRates();
  elements.speedSelect.replaceChildren(...rates.map((rate) => {
    const option = document.createElement("option");
    option.value = String(rate);
    option.textContent = `${rate}x`;
    option.selected = rate === 1;
    return option;
  }));
  elements.speedSelect.disabled = rates.length < 2;
}

function onPlayerReady() {
  state.mediaReady = true;
  elements.playButton.disabled = false;
  populatePlaybackRates();
  elements.seekSlider.max = String(playbackEnd());
  render();
}

function onPlayerError(event) {
  const messages = {
    2: "The YouTube video ID is invalid.", 5: "The YouTube player could not play this video.",
    100: "This YouTube video is unavailable or private.", 101: "The uploader does not allow embedding.",
    150: "The uploader does not allow embedding.", 153: "YouTube could not identify this player request.",
  };
  showError(`${messages[event.data] || "The YouTube video could not be loaded."} Use the Watch on YouTube link instead.`);
}

function loadYouTubeApi() {
  if (window.YT && window.YT.Player) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error("YouTube API timed out.")), 12000);
    window.onYouTubeIframeAPIReady = () => { window.clearTimeout(timeout); resolve(); };
    const script = document.createElement("script");
    script.src = "https://www.youtube.com/iframe_api";
    script.onerror = () => { window.clearTimeout(timeout); reject(new Error("YouTube API could not be loaded.")); };
    document.head.append(script);
  });
}

async function loadYouTubePlayer() {
  if (state.player || !state.data) return;
  elements.consentButton.disabled = true;
  elements.consentButton.textContent = "Loading YouTube...";
  try {
    await loadYouTubeApi();
    sessionStorage.setItem("youtube-consent", "granted");
    elements.consentPanel.hidden = true;
    elements.playerMount.hidden = false;
    state.player = new YT.Player("youtube-player", {
      videoId: state.data.videoId,
      host: "https://www.youtube-nocookie.com",
      playerVars: { autoplay: 0, controls: 1, origin: window.location.origin, playsinline: 1 },
      events: {
        onReady: onPlayerReady,
        onStateChange: (event) => { state.playerState = event.data; setPlayingState(); },
        onError: onPlayerError,
        onPlaybackRateChange: (event) => { elements.speedSelect.value = String(event.data); },
      },
    });
  } catch (error) {
    elements.consentButton.disabled = false;
    elements.consentButton.textContent = "Accept and retry YouTube";
    showError(`${error.message} Use the Watch on YouTube link instead.`);
  }
}

function installEvents() {
  elements.consentButton.addEventListener("click", loadYouTubePlayer);
  elements.playButton.addEventListener("click", togglePlayback);
  elements.cleanToggle.addEventListener("change", () => {
    state.showClean = elements.cleanToggle.checked;
    renderMetrics();
    rebuildMeters();
    rebuildLegend();
    render();
  });
  elements.speedSelect.addEventListener("change", () => {
    if (state.mediaReady) state.player.setPlaybackRate(Number(elements.speedSelect.value));
  });
  elements.seekSlider.addEventListener("pointerdown", () => { state.draggingSlider = true; beginScrub("slider"); });
  elements.seekSlider.addEventListener("input", () => seekTo(Number(elements.seekSlider.value)));
  const endSlider = () => { state.draggingSlider = false; endScrub("slider"); };
  window.addEventListener("pointerup", endSlider);
  window.addEventListener("pointercancel", endSlider);
  elements.chartResizeHandle.addEventListener("pointerdown", (event) => {
    state.resizingChart = true;
    state.chartResizeStartY = event.clientY;
    state.chartResizeStartHeight = elements.chartWrap.getBoundingClientRect().height;
    elements.chartResizeHandle.setPointerCapture(event.pointerId);
  });
  elements.chartResizeHandle.addEventListener("pointermove", (event) => {
    if (state.resizingChart) setChartHeight(state.chartResizeStartHeight + event.clientY - state.chartResizeStartY);
  });
  const endResize = () => { state.resizingChart = false; };
  elements.chartResizeHandle.addEventListener("pointerup", endResize);
  elements.chartResizeHandle.addEventListener("pointercancel", endResize);
  elements.chartResizeHandle.addEventListener("dblclick", () => { elements.chartWrap.style.removeProperty("--chart-height"); render(); });
  elements.chartWrap.tabIndex = 0;
  elements.chartWrap.addEventListener("pointerdown", (event) => {
    state.draggingChart = true; beginScrub("chart"); elements.chartWrap.setPointerCapture(event.pointerId); onChartPointer(event, true);
  });
  elements.chartWrap.addEventListener("pointermove", (event) => onChartPointer(event, state.draggingChart));
  elements.chartWrap.addEventListener("pointerup", (event) => { onChartPointer(event, true); state.draggingChart = false; endScrub("chart"); });
  elements.chartWrap.addEventListener("pointercancel", () => { state.draggingChart = false; endScrub("chart"); });
  elements.chartWrap.addEventListener("pointerleave", () => { if (!state.draggingChart) elements.tooltip.hidden = true; });
  elements.chartWrap.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault(); seekTo(currentPlaybackTime() + (event.key === "ArrowRight" ? 5 : -5));
    } else if (event.key === " " || event.key === "Enter") {
      event.preventDefault(); togglePlayback();
    }
  });
  window.addEventListener("resize", () => {
    cancelAnimationFrame(state.resizeFrame);
    state.resizeFrame = requestAnimationFrame(() => render());
  });
}

async function initialize() {
  installEvents();
  try {
    const response = await fetch(MANIFEST_URL);
    if (!response.ok) throw new Error(`manifest request returned ${response.status}`);
    state.manifest = await response.json();
    state.activeDemo = state.manifest.demos[0];
    await selectVideo(state.activeDemo.videos[0].id);
  } catch (error) {
    showError(`Could not initialize the demo: ${error.message}`);
    return;
  }
  if (sessionStorage.getItem("youtube-consent") === "granted") loadYouTubePlayer();
}

initialize();

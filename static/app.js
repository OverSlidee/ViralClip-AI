const urlInput = document.getElementById('url');
const ollamaModel = document.getElementById('ollamaModel');
const whisperModel = document.getElementById('whisperModel');
const clipCount = document.getElementById('clipCount');
const minClipDuration = document.getElementById('minClipDuration');
const maxClipDuration = document.getElementById('maxClipDuration');
const hookRecut = document.getElementById('hookRecut');
const hookDuration = document.getElementById('hookDuration');
const captionEnergy = document.getElementById('captionEnergy');
const captionPosition = document.getElementById('captionPosition');
const captionFontScale = document.getElementById('captionFontScale');
const captionMaxWords = document.getElementById('captionMaxWords');
const captionMarginV = document.getElementById('captionMarginV');
const captionTextCase = document.getElementById('captionTextCase');
const captionAnimation = document.getElementById('captionAnimation');
const captionKaraokeSpeed = document.getElementById('captionKaraokeSpeed');
const faceTracking = document.getElementById('faceTracking');
const brollEnabled = document.getElementById('brollEnabled');
const abTestMode = document.getElementById('abTestMode');
const brollOptions = document.getElementById('brollOptions');
const brollStyle = document.getElementById('brollStyle');
const pexelsApiKey = document.getElementById('pexelsApiKey');
const runBtn = document.getElementById('runBtn');
const progressList = document.getElementById('progressList');
const results = document.getElementById('results');
const ffmpegChip = document.getElementById('ffmpegChip');
const ollamaChip = document.getElementById('ollamaChip');
const templateGrid = document.getElementById('templateGrid');

let evt = null;
const PREFERRED_MODEL = 'qwen3.5:397b-cloud';
let statusPollTimer = null;
let activeJobId = null;

function modelLabel(name) {
  if (name === 'qwen3.5:397b-cloud') return 'qwen3.5:397b cloud';
  return name;
}

function addProgress(text) {
  const li = document.createElement('li');
  li.className = 'timeline-item';
  li.textContent = text;
  progressList.appendChild(li);
  progressList.scrollTop = progressList.scrollHeight;
}

function cleanupActiveTracking() {
  if (evt) {
    evt.close();
    evt = null;
  }
  if (statusPollTimer) {
    clearInterval(statusPollTimer);
    statusPollTimer = null;
  }
  activeJobId = null;
  runBtn.disabled = false;
}

function selectedTemplate() {
  const selected = document.querySelector('input[name="template"]:checked');
  return selected ? selected.value : 'viral_bold';
}

function currentBrollSource() {
  const selected = document.querySelector('input[name="brollSource"]:checked');
  return selected ? selected.value : 'local';
}

function renderResults(jobId, clips) {
  results.innerHTML = '';
  if (!clips.length) {
    results.innerHTML = '<p>No clips were generated.</p>';
    return;
  }

  clips.forEach((clip) => {
    const card = document.createElement('article');
    card.className = 'card';

    const variants = Array.isArray(clip.variant_outputs) ? clip.variant_outputs : [];
    const variantHtml = variants.length
      ? `
        <div class="variants">
          <h4>Variant Compare</h4>
          ${variants.map((v) => `
            <div class="variant-row">
              <span>${v.variant}</span>
              <span>score: ${v.score}</span>
              <a href="${v.preview_url}" target="_blank" rel="noopener">open</a>
            </div>
          `).join('')}
        </div>
      `
      : '';

    card.innerHTML = `
      <h3>${clip.title}</h3>
      <p>${clip.hook_reason}</p>
      <p><strong>Duration:</strong> ${clip.duration}s</p>
      <p><strong>Hook score:</strong> ${clip.hook_score || 'n/a'}</p>
      <p><strong>Hook recut:</strong> ${clip.hook_recut ? 'yes' : 'no'}</p>
      <p><strong>Montage:</strong> ${clip.montage_style || 'none'}</p>
      <p><strong>B-roll variant:</strong> ${clip.selected_broll_variant || 'n/a'}${clip.ab_test_mode ? ' (A/B auto)' : ''}</p>
      <video controls preload="metadata" src="${clip.preview_url}"></video>
      ${variantHtml}
      <a class="btn" href="/api/download/${jobId}/${clip.filename}">Download MP4</a>
    `;
    results.appendChild(card);
  });
}

function setChip(element, ok, text) {
  element.textContent = text;
  element.classList.remove('ok', 'bad');
  element.classList.add(ok ? 'ok' : 'bad');
}

async function loadHealth() {
  const resp = await fetch('/api/health');
  const data = await resp.json();
  setChip(ffmpegChip, !!data.dependencies?.ffmpeg?.ok, `FFmpeg: ${data.dependencies?.ffmpeg?.ok ? 'ready' : 'missing'}`);
  if (data.dependencies?.ollama?.ok) {
    setChip(ollamaChip, true, 'Ollama: ready');
    return;
  }

  if (data.dependencies?.ollama?.reachable) {
    setChip(ollamaChip, false, 'Ollama: no models');
  } else {
    setChip(ollamaChip, false, 'Ollama: offline');
  }
}

async function loadModels() {
  const resp = await fetch('/api/models');
  const data = await resp.json();
  ollamaModel.innerHTML = '';

  const models = data.models || [];
  if (!models.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = data.reachable ? 'No models installed' : 'Ollama offline';
    opt.selected = true;
    ollamaModel.appendChild(opt);
    runBtn.disabled = true;
    addProgress(data.reachable ? 'No Ollama models found. Run: ollama pull llama3.1' : 'Ollama is offline. Start Ollama and refresh.');
    return;
  }

  runBtn.disabled = false;
  const preferred = models.includes(PREFERRED_MODEL) ? PREFERRED_MODEL : (models.includes('llama3.1') ? 'llama3.1' : models[0]);
  models.forEach((name) => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = modelLabel(name);
    if (name === preferred) opt.selected = true;
    ollamaModel.appendChild(opt);
  });
}

brollEnabled.addEventListener('change', () => {
  brollOptions.classList.toggle('hidden', !brollEnabled.checked);
});

templateGrid.addEventListener('change', () => {
  document.querySelectorAll('.template-card').forEach((card) => {
    card.classList.toggle('active', card.dataset.template === selectedTemplate());
  });
});

runBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  if (!url) {
    alert('Please enter a YouTube URL.');
    return;
  }

  progressList.innerHTML = '';
  results.innerHTML = '';

  cleanupActiveTracking();

  runBtn.disabled = true;
  addProgress('Submitting job...');
  addProgress(`Caption position set to: ${captionPosition.value}`);

  const body = {
    url,
    ollama_model: ollamaModel.value,
    whisper_model: whisperModel.value,
    template: selectedTemplate(),
    clip_count: Number(clipCount.value || 4),
    min_clip_duration: Number(minClipDuration.value || 30),
    max_clip_duration: Number(maxClipDuration.value || 90),
    hook_recut: hookRecut.value === 'true',
    hook_duration: Number(hookDuration.value || 4),
    caption_energy: Number(captionEnergy.value || 1),
    caption_position: captionPosition.value,
    caption_font_scale: Number(captionFontScale.value || 1),
    caption_max_words_per_line: Number(captionMaxWords.value || 0),
    caption_margin_v: Number(captionMarginV.value || 0),
    caption_text_case: captionTextCase.value,
    caption_animation: captionAnimation.value,
    caption_karaoke_speed: Number(captionKaraokeSpeed.value || 1),
    face_tracking: faceTracking.checked,
    broll_enabled: brollEnabled.checked,
    broll_source: currentBrollSource(),
    broll_style: brollStyle.value,
    ab_test_mode: abTestMode.checked,
    pexels_api_key: pexelsApiKey.value.trim()
  };

  const startResp = await fetch('/api/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  if (!startResp.ok) {
    runBtn.disabled = false;
    let message = 'Failed to start processing job.';
    try {
      const err = await startResp.json();
      if (err.detail) message = err.detail;
    } catch (_) {
      // Keep generic message.
    }
    addProgress(message);
    return;
  }

  const { job_id: jobId } = await startResp.json();
  activeJobId = jobId;
  addProgress(`Job started: ${jobId}`);

  evt = new EventSource(`/api/progress/${jobId}`);

  // Fallback polling in case the SSE stream drops during long operations.
  statusPollTimer = setInterval(async () => {
    if (!activeJobId) return;
    try {
      const resp = await fetch(`/api/clips/${activeJobId}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.status === 'done' || data.status === 'error') {
        if (data.error) addProgress(`Error: ${data.error}`);
        renderResults(activeJobId, data.clips || []);
        cleanupActiveTracking();
      }
    } catch (_) {
      // Keep polling silently.
    }
  }, 5000);

  evt.addEventListener('progress', async (event) => {
    const data = JSON.parse(event.data);
    addProgress(`[${data.stage}] ${data.message}`);

    if (data.stage === 'done' || data.stage === 'error') {
      if (evt) {
        evt.close();
        evt = null;
      }

      const clipsResp = await fetch(`/api/clips/${jobId}`);
      const clipsData = await clipsResp.json();

      if (clipsData.error) {
        addProgress(`Error: ${clipsData.error}`);
      }

      renderResults(jobId, clipsData.clips || []);
      cleanupActiveTracking();
    }
  });

  evt.onerror = () => {
    addProgress('Progress stream interrupted, retrying...');
    // Do not close manually. EventSource auto-reconnects.
  };
});

loadModels().catch(() => {
  ollamaModel.innerHTML = '<option value="">Ollama offline</option>';
  runBtn.disabled = true;
});

loadHealth().catch(() => {
  setChip(ffmpegChip, false, 'FFmpeg: check failed');
  setChip(ollamaChip, false, 'Ollama: check failed');
});

/* ══════════════════════════════════════════════════════════════════
   Video Question Generator – Frontend Application
   ══════════════════════════════════════════════════════════════════ */

const API = '/api';

// ── State ──────────────────────────────────────────────────────────
let currentVideoId = null;
let allQuestions = [];

// ── DOM References ─────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressPercent = document.getElementById('progressPercent');
const progressSteps = document.getElementById('progressSteps');
const progressVideoName = document.getElementById('progressVideoName');
const statsBar = document.getElementById('statsBar');
const sceneTimeline = document.getElementById('sceneTimeline');
const timelineTrack = document.getElementById('timelineTrack');
const controlsBar = document.getElementById('controlsBar');
const questionsSection = document.getElementById('questionsSection');
const filterCategory = document.getElementById('filterCategory');
const filterDifficulty = document.getElementById('filterDifficulty');
const btnExportCSV = document.getElementById('btnExportCSV');
const btnExportJSON = document.getElementById('btnExportJSON');
const btnRegenerate = document.getElementById('btnRegenerate');

// Transcript DOM References
const transcriptSection = document.getElementById('transcriptSection');
const transcriptBody = document.getElementById('transcriptBody');
const transcriptText = document.getElementById('transcriptText');
const btnToggleTranscript = document.getElementById('btnToggleTranscript');
const btnDownloadTranscriptTxt = document.getElementById('btnDownloadTranscriptTxt');
const btnDownloadTranscriptJson = document.getElementById('btnDownloadTranscriptJson');

// Category colors
const CATEGORY_COLORS = {
    temporal: '#74b9ff',
    causal: '#a29bfe',
    counterfactual: '#fd79a8',
    contradiction: '#e17055',
    emotion: '#fdcb6e',
    multi_scene: '#00cec9',
    symbolic: '#e056fd',
    audio_visual_alignment: '#55efc4',
};

const CATEGORY_ICONS = {
    temporal: '⏱️',
    causal: '🔗',
    counterfactual: '🔀',
    contradiction: '⚡',
    emotion: '💭',
    multi_scene: '🎬',
    symbolic: '🔮',
    audio_visual_alignment: '🖼️',
};

const STEP_LABELS = {
    scene_segmentation: 'Scene Segmentation',
    frame_extraction: 'Frame Extraction',
    audio_processing: 'Audio Processing',
    video_understanding: 'Video Understanding',
    graph_construction: 'Graph Construction',
    memory_indexing: 'Memory Indexing',
    question_generation: 'Question Generation',
    difficulty_estimation: 'Difficulty Estimation',
    uniqueness_filter: 'Uniqueness Filter',
    saving: 'Saving Results',
    completed: 'Completed',
    error: 'Error',
};


// ══════════════════════════════════════════════════════════════════
// Upload Handling
// ══════════════════════════════════════════════════════════════════

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

async function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.mp4')) {
        showToast('Please select an MP4 file', 'error');
        return;
    }

    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > 500) {
        showToast(`File too large (${sizeMB.toFixed(1)} MB). Max: 500 MB`, 'error');
        return;
    }

    try {
        // Upload
        uploadZone.querySelector('.label').textContent = 'Uploading…';
        uploadZone.querySelector('.hint').textContent = `${file.name} (${sizeMB.toFixed(1)} MB)`;

        const language = document.getElementById('languageSelect').value;
        const formData = new FormData();
        formData.append('file', file);
        formData.append('language', language);

        const uploadResp = await fetch(`${API}/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!uploadResp.ok) {
            const err = await uploadResp.json().catch(() => ({}));
            throw new Error(err.detail || 'Upload failed');
        }

        const uploadData = await uploadResp.json();
        currentVideoId = uploadData.video.id;

        // Reset & show progress
        progressVideoName.textContent = file.name;
        showProgress();

        // Start processing
        const processResp = await fetch(`${API}/process/${currentVideoId}`, {
            method: 'POST',
        });

        if (!processResp.ok) {
            throw new Error('Failed to start processing');
        }

        // Listen for progress events
        listenProgress(currentVideoId);

    } catch (err) {
        showToast(err.message, 'error');
        resetUploadZone();
    }
}

function resetUploadZone() {
    uploadZone.querySelector('.label').textContent = 'Drop your MP4 video here or click to browse';
    uploadZone.querySelector('.hint').textContent = 'Supports MP4 files up to 500 MB';
    fileInput.value = '';
}


// ══════════════════════════════════════════════════════════════════
// Progress Streaming (SSE)
// ══════════════════════════════════════════════════════════════════

function showProgress() {
    progressSection.classList.add('active');
    progressBar.style.width = '0%';
    progressPercent.textContent = '0%';
    progressSteps.innerHTML = '';

    // Hide results until ready
    statsBar.classList.remove('active');
    sceneTimeline.classList.remove('active');
    controlsBar.classList.remove('active');
    questionsSection.classList.remove('active');
    transcriptSection.style.display = 'none';
}

function listenProgress(videoId) {
    const evtSource = new EventSource(`${API}/progress/${videoId}`);
    const seenSteps = new Set();

    evtSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const { step, percent, detail } = data;

        if (percent >= 0) {
            progressBar.style.width = `${percent}%`;
            progressPercent.textContent = `${percent}%`;
        }

        // Add step entry
        if (!seenSteps.has(step) || step === 'completed' || step === 'error') {
            seenSteps.add(step);
            const stepEl = document.createElement('div');
            stepEl.className = 'progress-step';

            let iconClass = 'active';
            let iconSymbol = '●';
            if (step === 'completed') { iconClass = 'done'; iconSymbol = '✓'; }
            if (step === 'error') { iconClass = 'error'; iconSymbol = '✗'; }

            stepEl.innerHTML = `
                <span class="step-icon ${iconClass}">${iconSymbol}</span>
                <span>${STEP_LABELS[step] || step}: ${detail}</span>
            `;
            progressSteps.appendChild(stepEl);
            progressSteps.scrollTop = progressSteps.scrollHeight;

            // Mark previous steps as done
            const prevIcons = progressSteps.querySelectorAll('.step-icon.active');
            prevIcons.forEach((el, i) => {
                if (i < prevIcons.length - 1) {
                    el.className = 'step-icon done';
                    el.textContent = '✓';
                }
            });
        }

        // Pipeline finished
        if (step === 'completed') {
            evtSource.close();
            setTimeout(() => loadResults(videoId), 800);
        }

        if (step === 'error') {
            evtSource.close();
            showToast(`Processing failed: ${detail}`, 'error');
        }
    };

    evtSource.onerror = () => {
        evtSource.close();
        // Check if processing completed while we were disconnected
        setTimeout(() => checkAndLoadResults(videoId), 2000);
    };
}

async function checkAndLoadResults(videoId) {
    try {
        const resp = await fetch(`${API}/videos/${videoId}`);
        const data = await resp.json();
        if (data.video && data.video.status === 'completed') {
            loadResults(videoId);
        }
    } catch (e) { /* silent */ }
}


// ══════════════════════════════════════════════════════════════════
// Load & Render Results
// ══════════════════════════════════════════════════════════════════

async function loadResults(videoId) {
    try {
        const [questionsResp, scenesResp, transcriptResp] = await Promise.all([
            fetch(`${API}/questions/${videoId}`),
            fetch(`${API}/scenes/${videoId}`),
            fetch(`${API}/transcript/${videoId}`).catch(() => null),
        ]);

        const questionsData = await questionsResp.json();
        const scenesData = await scenesResp.json();

        let transcriptData = null;
        if (transcriptResp && transcriptResp.ok) {
            try {
                transcriptData = await transcriptResp.json();
            } catch (e) { }
        }

        allQuestions = questionsData.questions || [];
        const scenes = scenesData.scenes || [];

        // Stats
        renderStats(allQuestions, scenes);

        // Scene timeline
        renderTimeline(scenes);

        // Render Transcript
        renderTranscript(transcriptData);

        // Controls
        controlsBar.classList.add('active');

        // Questions
        renderQuestions(allQuestions);

        // Hide progress after a moment
        setTimeout(() => {
            progressSection.classList.remove('active');
        }, 500);

        // Reset upload zone for next video
        resetUploadZone();

        showToast(`✨ Generated ${allQuestions.length} questions!`, 'success');

    } catch (err) {
        showToast('Failed to load results: ' + err.message, 'error');
    }
}

function renderStats(questions, scenes) {
    const totalQ = questions.length;
    const expertCount = questions.filter(q => q.difficulty === 'expert').length;
    const categories = new Set(questions.map(q => q.category));

    document.getElementById('statTotal').textContent = totalQ;
    document.getElementById('statScenes').textContent = scenes.length;
    document.getElementById('statCategories').textContent = categories.size;
    document.getElementById('statExpert').textContent = expertCount;

    statsBar.classList.add('active');
}

function renderTimeline(scenes) {
    if (!scenes.length) return;

    const totalDuration = scenes.reduce((max, s) => Math.max(max, s.end_time || 0), 0);
    timelineTrack.innerHTML = '';

    const colors = ['#6c5ce7', '#a29bfe', '#74b9ff', '#00cec9', '#55efc4',
        '#fdcb6e', '#e17055', '#fd79a8', '#e056fd', '#636e72'];

    scenes.forEach((scene, i) => {
        const duration = (scene.end_time || 0) - (scene.start_time || 0);
        const widthPct = totalDuration > 0 ? (duration / totalDuration * 100) : (100 / scenes.length);

        const seg = document.createElement('div');
        seg.className = 'timeline-segment';
        seg.style.width = `${Math.max(widthPct, 2)}%`;
        seg.style.background = colors[i % colors.length];
        seg.textContent = `S${i + 1}`;
        seg.innerHTML += `
            <div class="tooltip">
                Scene ${i + 1}<br>
                ${formatTime(scene.start_time)} – ${formatTime(scene.end_time)}<br>
                ${scene.summary ? scene.summary.substring(0, 80) + '…' : ''}
            </div>
        `;

        timelineTrack.appendChild(seg);
    });

    sceneTimeline.classList.add('active');
}

function renderQuestions(questions) {
    questionsSection.innerHTML = '';

    if (!questions.length) {
        questionsSection.innerHTML = `
            <div class="empty-state">
                <div class="icon">🤔</div>
                <div class="title">No questions match the current filters</div>
            </div>
        `;
        questionsSection.classList.add('active');
        return;
    }

    // Group by category
    const grouped = {};
    questions.forEach(q => {
        const cat = q.category || 'other';
        if (!grouped[cat]) grouped[cat] = [];
        grouped[cat].push(q);
    });

    // Render each category group
    Object.entries(grouped).forEach(([category, catQuestions]) => {
        const group = document.createElement('div');
        group.className = 'category-group';

        const color = CATEGORY_COLORS[category] || '#9898b0';
        const icon = CATEGORY_ICONS[category] || '❓';

        group.innerHTML = `
            <div class="category-header">
                <span class="category-dot" style="background: ${color}"></span>
                <span class="category-name">${icon} ${formatCategoryName(category)}</span>
                <span class="category-count">${catQuestions.length} questions</span>
            </div>
            <div class="questions-grid" id="grid-${category}"></div>
        `;

        questionsSection.appendChild(group);

        const grid = group.querySelector('.questions-grid');

        catQuestions.forEach((q, idx) => {
            const card = createQuestionCard(q, idx, color);
            grid.appendChild(card);
        });
    });

    questionsSection.classList.add('active');
}

function createQuestionCard(q, idx, accentColor) {
    const card = document.createElement('div');
    card.className = 'question-card';
    card.style.setProperty('--card-accent', accentColor);

    const diffClass = (q.difficulty || 'medium').toLowerCase();
    const personaText = q.persona ? `${formatPersona(q.persona)} perspective` : '';
    const cardId = `answer-${q.id || idx}`;
    const options = Array.isArray(q.mc_options) ? q.mc_options : [];
    const correctIndex = Number.isInteger(q.correct_option) ? q.correct_option : 0;
    const explanation = q.explanation || q.answer_text || '';
    const visualRefs = Array.isArray(q.visual_refs) ? q.visual_refs : [];

    card.innerHTML = `
        <div class="card-top">
            <span class="persona-badge">${personaText}</span>
            <span class="difficulty-badge ${diffClass}">${q.difficulty || 'N/A'}</span>
        </div>
        <div class="question-text">${escapeHtml(q.question_text)}</div>
        ${options.length ? createMcqOptionsHtml(options, correctIndex) : ''}
        ${explanation ? `
            <div class="answer-actions">
                <button class="answer-toggle" onclick="toggleAnswer('${cardId}', this)">
                    ▶ Show Explanation
                </button>
                ${q.audio_path ? `
                    <button class="btn-audio" onclick="playExplanation('${q.audio_path}', this)" title="Listen to AI Explanation">
                        🔊 Listen
                    </button>
                ` : ''}
            </div>
            <div class="answer-content" id="${cardId}">
                <div class="answer-label">Correct answer: ${escapeHtml(formatOptionLabel(correctIndex, options[correctIndex]))}</div>
                <div>${escapeHtml(explanation)}</div>
            </div>
        ` : ''}
        <div class="card-meta">
            ${q.difficulty_score != null ? `<span class="meta-item">📊 Score: ${q.difficulty_score}</span>` : ''}
            ${q.novelty_score != null ? `<span class="meta-item">✨ Novelty: ${q.novelty_score}</span>` : ''}
            ${q.scenes_involved ? `<span class="meta-item">🎬 ${Array.isArray(q.scenes_involved) ? q.scenes_involved.length : '?'} scenes</span>` : ''}
        </div>
    `;

    return card;
}

function createMcqOptionsHtml(options, correctIndex) {
    return `
        <div class="mcq-options">
            ${options.map((option, index) => `
                <div class="mcq-option">
                    <span class="mcq-letter">${String.fromCharCode(65 + index)}</span>
                    <span>${escapeHtml(option)}</span>
                </div>
            `).join('')}
        </div>
    `;
}

// Visual references removed: questions are now transcript-focused and do not include generated images.

function formatOptionLabel(index, option) {
    const letter = String.fromCharCode(65 + Math.max(0, index || 0));
    return option ? `${letter}. ${option}` : letter;
}


// ══════════════════════════════════════════════════════════════════
// Filters
// ══════════════════════════════════════════════════════════════════

filterCategory.addEventListener('change', applyFilters);
filterDifficulty.addEventListener('change', applyFilters);

function applyFilters() {
    const cat = filterCategory.value;
    const diff = filterDifficulty.value;

    let filtered = allQuestions;
    if (cat) filtered = filtered.filter(q => q.category === cat);
    if (diff) filtered = filtered.filter(q => q.difficulty === diff);

    renderQuestions(filtered);
}


// ══════════════════════════════════════════════════════════════════
// Export
// ══════════════════════════════════════════════════════════════════

btnExportCSV.addEventListener('click', () => {
    if (!currentVideoId) return;
    const cat = filterCategory.value;
    const diff = filterDifficulty.value;
    let url = `${API}/export/${currentVideoId}?format=csv`;
    if (cat) url += `&category=${cat}`;
    if (diff) url += `&difficulty=${diff}`;
    window.open(url, '_blank');
});

btnExportJSON.addEventListener('click', () => {
    if (!currentVideoId) return;
    const cat = filterCategory.value;
    const diff = filterDifficulty.value;
    let url = `${API}/export/${currentVideoId}?format=json`;
    if (cat) url += `&category=${cat}`;
    if (diff) url += `&difficulty=${diff}`;
    window.open(url, '_blank');
});


// ══════════════════════════════════════════════════════════════════
// Regenerate
// ══════════════════════════════════════════════════════════════════

btnRegenerate.addEventListener('click', async () => {
    if (!currentVideoId) return;
    if (!confirm('Regenerate all questions? This will delete the existing ones.')) return;

    try {
        const cat = filterCategory.value;
        let url = `${API}/regenerate/${currentVideoId}`;
        if (cat) url += `?category=${cat}`;

        const resp = await fetch(url, { method: 'POST' });
        if (!resp.ok) throw new Error('Regeneration failed');

        showProgress();
        progressVideoName.textContent = 'regenerating…';
        listenProgress(currentVideoId);

    } catch (err) {
        showToast(err.message, 'error');
    }
});


// ══════════════════════════════════════════════════════════════════
// Utilities
// ══════════════════════════════════════════════════════════════════

function toggleAnswer(id, btn) {
    const el = document.getElementById(id);
    if (!el) return;
    const isVisible = el.classList.toggle('visible');
    btn.textContent = isVisible ? '▼ Hide Explanation' : '▶ Show Explanation';
}

function formatTime(seconds) {
    if (seconds == null) return '0:00';
    const totalSeconds = Math.max(0, Math.floor(seconds));
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60;

    if (h > 0) {
        return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatCategoryName(cat) {
    return cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatPersona(persona) {
    return persona.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(60px)';
        toast.style.transition = 'all 0.4s ease';
        setTimeout(() => toast.remove(), 400);
    }, 4000);
}

// ── Transcript Rendering & Interactivity ────────────────────────────

function renderTranscript(transcriptData) {
    if (!transcriptData || !transcriptData.transcript) {
        transcriptSection.style.display = 'none';
        return;
    }

    transcriptSection.style.display = 'block';
    transcriptBody.classList.remove('collapsed');
    btnToggleTranscript.innerText = '▼ Collapse';
    transcriptText.innerHTML = '';

    const transcriptObj = transcriptData.transcript;
    const segments = transcriptObj.segments || [];
    const hasPlainText = Boolean((transcriptObj.text || '').trim());

    btnDownloadTranscriptTxt.disabled = !segments.length && !hasPlainText;
    btnDownloadTranscriptJson.disabled = false;

    if (segments.length === 0 && hasPlainText) {
        const p = document.createElement('div');
        p.className = 'transcript-text-content';
        p.innerText = transcriptObj.text;
        transcriptText.appendChild(p);
    } else if (segments.length > 0) {
        segments.forEach(seg => {
            const div = document.createElement('div');
            div.className = 'transcript-segment';

            const spanTime = document.createElement('span');
            spanTime.className = 'transcript-timestamp';
            spanTime.innerText = formatTime(seg.start);

            const spanText = document.createElement('span');
            spanText.className = 'transcript-text-content';
            spanText.innerText = seg.text;

            div.appendChild(spanTime);
            div.appendChild(spanText);
            transcriptText.appendChild(div);
        });
    } else {
        const empty = document.createElement('div');
        empty.className = 'transcript-empty';

        let message = 'No spoken words were detected in this video.';
        if (transcriptObj.status === 'unavailable') {
            message = 'Audio extraction failed. Please try uploading the video again.';
        } else if (transcriptObj.status === 'empty') {
            message = 'Transcription completed but found no speech in the audio.';
        } else if (transcriptObj.status === 'reused') {
            message = 'Using cached transcript from a previous upload of the same video.';
        } else if (transcriptObj.error) {
            message = `Transcription error: ${transcriptObj.error}`;
        }

        empty.textContent = message;
        transcriptText.appendChild(empty);
    }
}

// Event Listeners for Transcript Actions
if (btnToggleTranscript) {
    btnToggleTranscript.addEventListener('click', () => {
        transcriptBody.classList.toggle('collapsed');
        if (transcriptBody.classList.contains('collapsed')) {
            btnToggleTranscript.innerText = '▲ Expand';
        } else {
            btnToggleTranscript.innerText = '▼ Collapse';
        }
    });
}

if (btnDownloadTranscriptTxt) {
    btnDownloadTranscriptTxt.addEventListener('click', () => {
        if (currentVideoId) {
            window.open(`${API}/export-transcript/${currentVideoId}?format=txt`, '_blank');
        }
    });
}

if (btnDownloadTranscriptJson) {
    btnDownloadTranscriptJson.addEventListener('click', () => {
        if (currentVideoId) {
            window.open(`${API}/export-transcript/${currentVideoId}?format=json`, '_blank');
        }
    });
}

// Audio Explanation Player
let currentAudio = null;
let currentAudioBtn = null;

function playExplanation(path, btn) {
    if (currentAudio) {
        currentAudio.pause();
        if (currentAudioBtn) {
            currentAudioBtn.innerHTML = '🔊 Listen';
            currentAudioBtn.classList.remove('playing');
        }

        // If clicking the same button, just stop
        if (currentAudioBtn === btn) {
            currentAudio = null;
            currentAudioBtn = null;
            return;
        }
    }

    currentAudio = new Audio(path);
    currentAudioBtn = btn;

    btn.innerHTML = '⏸ Playing…';
    btn.classList.add('playing');

    currentAudio.play().catch(err => {
        showToast('Playback failed: ' + err.message, 'error');
        btn.innerHTML = '🔊 Listen';
        btn.classList.remove('playing');
    });

    currentAudio.onended = () => {
        btn.innerHTML = '🔊 Listen';
        btn.classList.remove('playing');
        currentAudio = null;
        currentAudioBtn = null;
    };
}

// Make functions globally available
window.toggleAnswer = toggleAnswer;
window.playExplanation = playExplanation;

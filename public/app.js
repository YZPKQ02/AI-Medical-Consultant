const state = {
  consultationId: null,
  isLoading: false
};

const elements = {
  form: document.querySelector("#consultationForm"),
  input: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  newButton: document.querySelector("#newConsultationButton"),
  messageList: document.querySelector("#messageList"),
  urgencyBadge: document.querySelector("#urgencyBadge"),
  departmentValue: document.querySelector("#departmentValue"),
  symptomValue: document.querySelector("#symptomValue"),
  urgentValue: document.querySelector("#urgentValue"),
  sourceList: document.querySelector("#sourceList"),
  ageInput: document.querySelector("#ageInput"),
  genderInput: document.querySelector("#genderInput"),
  allergyInput: document.querySelector("#allergyInput"),
  chronicInput: document.querySelector("#chronicInput"),
  profileStatus: document.querySelector("#profileStatus"),
  startWithProfileButton: document.querySelector("#startWithProfileButton"),
  refreshHistoryButton: document.querySelector("#refreshHistoryButton"),
  consultationList: document.querySelector("#consultationList"),
  knowledgeSearchForm: document.querySelector("#knowledgeSearchForm"),
  knowledgeQueryInput: document.querySelector("#knowledgeQueryInput"),
  knowledgeResults: document.querySelector("#knowledgeResults")
};

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = elements.input.value.trim();
  if (!content || state.isLoading) {
    return;
  }

  setLoading(true);
  appendMessage("user", content);
  elements.input.value = "";

  try {
    await ensureConsultation(content);
    const response = await api(`/api/consultations/${state.consultationId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content })
    });

    appendMessage("assistant", response.assistant_message.content);
    renderTriage(response.assistant_message.analysis);
    renderSources(response.assistant_message.source_knowledge);
    await refreshConsultations();
  } catch (error) {
    appendMessage("assistant", `请求失败：${error.message}`);
  } finally {
    setLoading(false);
  }
});

elements.newButton.addEventListener("click", () => {
  resetWorkspace({
    message: "已开启新的问诊。请描述主要症状、持续时间、伴随表现和既往病史。"
  });
});

elements.startWithProfileButton.addEventListener("click", async () => {
  if (state.isLoading) {
    return;
  }

  setLoading(true);
  try {
    const profile = collectUserContext();
    const response = await api("/api/consultations", {
      method: "POST",
      body: JSON.stringify({
        chief_complaint: "待补充",
        user_context: profile
      })
    });
    state.consultationId = response.id;
    resetWorkspace({
      keepSession: true,
      message: "已根据左侧患者信息建立问诊档案。现在可以继续描述症状。"
    });
    renderProfileStatus("已应用", "low");
    await refreshConsultations();
  } catch (error) {
    renderProfileStatus("建档失败", "high");
    appendMessage("assistant", `建档失败：${error.message}`);
  } finally {
    setLoading(false);
  }
});

elements.refreshHistoryButton.addEventListener("click", refreshConsultations);

elements.knowledgeSearchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = elements.knowledgeQueryInput.value.trim();
  await searchKnowledge(query);
});

elements.knowledgeQueryInput.addEventListener("input", () => {
  if (!elements.knowledgeQueryInput.value.trim()) {
    clearKnowledgeResults();
  }
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.isComposing) {
    return;
  }

  if (event.ctrlKey && event.shiftKey) {
    insertTextAtCursor(elements.input, "\n");
    event.preventDefault();
    return;
  }

  event.preventDefault();
  elements.form.requestSubmit();
});

for (const input of [
  elements.ageInput,
  elements.genderInput,
  elements.allergyInput,
  elements.chronicInput
]) {
  input.addEventListener("input", () => {
    renderProfileStatus(state.consultationId ? "下次生效" : "待应用", "neutral");
  });
}

refreshConsultations();

async function ensureConsultation(chiefComplaint) {
  if (state.consultationId) {
    return;
  }

  const response = await api("/api/consultations", {
    method: "POST",
    body: JSON.stringify({
      chief_complaint: chiefComplaint,
      user_context: collectUserContext()
    })
  });
  state.consultationId = response.id;
  renderProfileStatus("已应用", "low");
  await refreshConsultations();
}

async function refreshConsultations() {
  try {
    const response = await api("/api/consultations");
    renderConsultations(response.consultations || []);
  } catch (error) {
    elements.consultationList.innerHTML = "";
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = `会话加载失败：${error.message}`;
    elements.consultationList.append(empty);
  }
}

async function openConsultation(consultationId) {
  if (state.isLoading) {
    return;
  }

  try {
    const consultation = await api(`/api/consultations/${consultationId}`);
    state.consultationId = consultation.id;
    hydrateProfile(consultation.user_context || {});
    renderProfileStatus("已应用", "low");
    elements.messageList.innerHTML = "";

    if (!consultation.messages || consultation.messages.length === 0) {
      appendMessage("assistant", "已打开这个问诊档案，但还没有对话记录。");
      renderTriage(null);
      renderSources([]);
    } else {
      for (const message of consultation.messages) {
        appendMessage(message.role, message.content);
      }
      const lastAssistant = [...consultation.messages].reverse().find((message) => message.role === "assistant");
      renderTriage(lastAssistant?.analysis || null);
      renderSources(lastAssistant?.source_knowledge || []);
    }

    renderConsultationsSelection();
  } catch (error) {
    appendMessage("assistant", `打开会话失败：${error.message}`);
  }
}

async function deleteConsultation(consultationId, title) {
  if (state.isLoading) {
    return;
  }

  const confirmed = window.confirm(`确定删除“${title || "未命名问诊"}”吗？删除后当前会话记录将不再显示。`);
  if (!confirmed) {
    return;
  }

  try {
    await api(`/api/consultations/${consultationId}`, { method: "DELETE" });
    if (state.consultationId === consultationId) {
      resetWorkspace({
        message: "已删除当前问诊。可以从这里开始新的问诊。"
      });
    }
    await refreshConsultations();
  } catch (error) {
    appendMessage("assistant", `删除会话失败：${error.message}`);
  }
}

async function searchKnowledge(query) {
  if (!query) {
    clearKnowledgeResults();
    return;
  }

  elements.knowledgeResults.innerHTML = "";
  const loading = document.createElement("p");
  loading.className = "muted";
  loading.textContent = "正在检索知识库...";
  elements.knowledgeResults.append(loading);

  try {
    const path = `/api/knowledge?q=${encodeURIComponent(query)}`;
    const response = await api(path);
    renderKnowledgeResults(response.results || []);
  } catch (error) {
    elements.knowledgeResults.innerHTML = "";
    const failed = document.createElement("p");
    failed.className = "muted";
    failed.textContent = `检索失败：${error.message}`;
    elements.knowledgeResults.append(failed);
  }
}

function clearKnowledgeResults() {
  elements.knowledgeResults.innerHTML = "";
  const empty = document.createElement("p");
  empty.className = "muted";
  empty.textContent = "输入关键词后检索知识库。";
  elements.knowledgeResults.append(empty);
}

function insertTextAtCursor(input, text) {
  const start = input.selectionStart;
  const end = input.selectionEnd;
  const before = input.value.slice(0, start);
  const after = input.value.slice(end);
  input.value = `${before}${text}${after}`;
  const cursor = start + text.length;
  input.selectionStart = cursor;
  input.selectionEnd = cursor;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "content-type": "application/json",
      ...(options.headers ?? {})
    },
    ...options
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || payload.detail || "接口请求失败");
  }
  return payload;
}

function collectUserContext() {
  return {
    age: elements.ageInput.value.trim(),
    gender: elements.genderInput.value,
    allergies: elements.allergyInput.value.trim(),
    chronic_diseases: elements.chronicInput.value.trim()
  };
}

function hydrateProfile(profile) {
  elements.ageInput.value = profile.age || "";
  elements.genderInput.value = profile.gender || "";
  elements.allergyInput.value = profile.allergies || "";
  elements.chronicInput.value = profile.chronic_diseases || "";
}

function resetWorkspace({ message, keepSession = false }) {
  if (!keepSession) {
    state.consultationId = null;
    renderProfileStatus("待应用", "neutral");
  }
  elements.messageList.innerHTML = "";
  appendMessage("assistant", message);
  renderTriage(null);
  renderSources([]);
  renderConsultationsSelection();
}

function appendMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  article.append(avatar, bubble);
  elements.messageList.append(article);
  elements.messageList.scrollTop = elements.messageList.scrollHeight;
}

function renderTriage(analysis) {
  if (!analysis) {
    elements.urgencyBadge.className = "badge neutral";
    elements.urgencyBadge.textContent = "未评估";
    elements.departmentValue.textContent = "等待问诊";
    elements.symptomValue.textContent = "暂无";
    elements.urgentValue.textContent = "暂无";
    return;
  }

  const urgencyClass = analysis.urgency_level >= 4 ? "high" : analysis.urgency_level >= 2 ? "medium" : "low";
  elements.urgencyBadge.className = `badge ${urgencyClass}`;
  elements.urgencyBadge.textContent = `${analysis.urgency_level}/4`;
  elements.departmentValue.textContent = analysis.department;
  elements.symptomValue.textContent = analysis.symptoms.length ? analysis.symptoms.join("、") : "待补充";
  elements.urgentValue.textContent = analysis.needs_urgent_care ? "建议立即就医" : "暂未识别急症";
}

function renderSources(sources) {
  renderSourceItems(elements.sourceList, sources, "暂无匹配知识来源。");
}

function renderKnowledgeResults(results) {
  renderSourceItems(elements.knowledgeResults, results, "未检索到匹配知识。");
}

function renderSourceItems(container, sources, emptyText) {
  container.innerHTML = "";

  if (!sources || sources.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = emptyText;
    container.append(empty);
    return;
  }

  for (const source of sources) {
    const item = document.createElement("article");
    item.className = "source-item";

    const title = document.createElement("strong");
    title.textContent = source.relevance
      ? `${source.title} · 相关度 ${source.relevance}`
      : source.title;

    const meta = document.createElement("span");
    meta.className = "source-meta";
    meta.textContent = source.department || source.retrieval_reason || "";

    const content = document.createElement("p");
    content.textContent = source.content;

    item.append(title, meta, content);
    container.append(item);
  }
}

function renderConsultations(consultations) {
  elements.consultationList.innerHTML = "";

  if (consultations.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "还没有问诊记录。";
    elements.consultationList.append(empty);
    return;
  }

  for (const consultation of consultations.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "history-item";
    row.dataset.consultationId = consultation.id;

    const button = document.createElement("button");
    button.className = "history-button";
    button.type = "button";
    button.addEventListener("click", () => openConsultation(consultation.id));

    const title = document.createElement("strong");
    title.textContent = consultation.chief_complaint || "未命名问诊";

    const meta = document.createElement("span");
    meta.textContent = `${consultation.message_count} 条消息 · ${formatTime(consultation.updated_at)}`;

    const deleteButton = document.createElement("button");
    deleteButton.className = "history-delete-button";
    deleteButton.type = "button";
    deleteButton.setAttribute("aria-label", `删除${title.textContent}`);
    deleteButton.textContent = "删除";
    deleteButton.addEventListener("click", () => deleteConsultation(consultation.id, title.textContent));

    button.append(title, meta);
    row.append(button, deleteButton);
    elements.consultationList.append(row);
  }

  renderConsultationsSelection();
}

function renderConsultationsSelection() {
  for (const item of elements.consultationList.querySelectorAll(".history-item")) {
    item.classList.toggle("active", item.dataset.consultationId === state.consultationId);
  }
}

function renderProfileStatus(label, level) {
  elements.profileStatus.className = `badge ${level}`;
  elements.profileStatus.textContent = label;
}

function formatTime(value) {
  if (!value) {
    return "刚刚";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function setLoading(value) {
  state.isLoading = value;
  elements.sendButton.disabled = value;
  elements.startWithProfileButton.disabled = value;
  elements.sendButton.textContent = value ? "分析中..." : "发送";
}

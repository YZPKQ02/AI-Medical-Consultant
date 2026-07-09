import { readFile } from "node:fs/promises";
import test from "node:test";
import assert from "node:assert/strict";

test("frontend has consultation form and API integration", async () => {
  const html = await readFile("public/index.html", "utf8");
  const appJs = await readFile("public/app.js", "utf8");

  assert.match(html, /id="consultationForm"/);
  assert.match(html, /id="messageInput"/);
  assert.match(html, /id="consultationList"/);
  assert.match(html, /id="knowledgeSearchForm"/);
  assert.match(appJs, /\/api\/consultations/);
  assert.match(appJs, /refreshConsultations/);
  assert.match(appJs, /deleteConsultation/);
  assert.match(appJs, /method: "DELETE"/);
  assert.match(appJs, /clearKnowledgeResults/);
  assert.match(appJs, /requestSubmit/);
  assert.match(appJs, /source_knowledge/);
});

test("frontend renders triage and source panels", async () => {
  const html = await readFile("public/index.html", "utf8");

  assert.match(html, /当前分诊/);
  assert.match(html, /知识来源/);
  assert.match(html, /会话历史/);
  assert.match(html, /知识库检索/);
  assert.match(html, /AI Medical Consultant/);
});

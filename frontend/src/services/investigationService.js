import { api } from "./api";

export async function startInvestigation(caseId) {
  const { data } = await api.post(`/investigation/start/${caseId}`);
  return data;
}

export async function askQuestion(sessionId, question) {
  const { data } = await api.post(`/investigation/${sessionId}/ask`, { question });
  return data;
}

export async function getHistory(sessionId) {
  const { data } = await api.get(`/investigation/${sessionId}/history`);
  return data;
}

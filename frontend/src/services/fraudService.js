import { api } from "./api";

export async function analyzeFraud(invoiceId) {
  const { data } = await api.post(`/fraud/analyze/${invoiceId}`);
  return data;
}

export async function listCases() {
  const { data } = await api.get("/fraud/cases");
  return data;
}

export async function getCase(id) {
  const { data } = await api.get(`/fraud/cases/${id}`);
  return data;
}

export async function updateDecision(id, payload) {
  const { data } = await api.patch(`/fraud/cases/${id}`, payload);
  return data;
}

export async function generateSAR(id) {
  const { data } = await api.post(`/fraud/cases/${id}/sar`);
  return data;
}

export async function getDashboardSummary() {
  const { data } = await api.get("/dashboard/summary");
  return data;
}

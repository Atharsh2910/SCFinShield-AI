import { api } from "./api";

export async function generateDataset(config) {
  const { data } = await api.post("/simulator/generate", config);
  return data;
}

export async function runScenario(config) {
  const { data } = await api.post("/simulator/scenario", config);
  return data;
}

export async function listScenarioTemplates() {
  const { data } = await api.get("/simulator/scenarios");
  return data;
}

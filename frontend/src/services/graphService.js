import { api } from "./api";

export async function getEntityNetwork(id, depth = 2) {
  const { data } = await api.get(`/graph/network/${id}`, { params: { depth } });
  return data;
}

export async function detectCarousel(entityId) {
  const { data } = await api.get(`/graph/carousel/${entityId}`);
  return data;
}

export async function traceCascade(invoiceId) {
  const { data } = await api.get(`/graph/cascade/${invoiceId}`);
  return data;
}

export async function getConcentrationRisk(lenderId) {
  const { data } = await api.get(`/graph/concentration/${lenderId}`);
  return data;
}

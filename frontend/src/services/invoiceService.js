import { api } from "./api";

export async function uploadInvoice(file, lenderName = "") {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("lender_name", lenderName);
  const { data } = await api.post("/invoices/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function analyzeInvoice(id) {
  const { data } = await api.post(`/invoices/analyze/${id}`);
  return data;
}

export async function listInvoices(filters = {}) {
  const { data } = await api.get("/invoices/", { params: filters });
  return data;
}

export async function getInvoice(id) {
  const { data } = await api.get(`/invoices/${id}`);
  return data;
}

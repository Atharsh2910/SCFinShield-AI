import { create } from "zustand";

export const useFraudStore = create((set) => ({
  activeInvoices: [],
  selectedCase: null,
  selectedInvoiceIds: [],
  investigationSessionId: null,
  dashboardFilters: {},
  setActiveInvoices: (activeInvoices) => set({ activeInvoices }),
  setSelectedCase: (selectedCase) => set({ selectedCase }),
  setSelectedInvoiceIds: (selectedInvoiceIds) => set({ selectedInvoiceIds }),
  setInvestigationSessionId: (investigationSessionId) => set({ investigationSessionId }),
  setDashboardFilters: (dashboardFilters) => set({ dashboardFilters }),
}));

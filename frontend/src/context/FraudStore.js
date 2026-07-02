import { create } from "zustand";

export const useFraudStore = create((set) => ({
  selectedCase: null,
  selectedInvoiceIds: [],
  investigationSessionId: null,
  dashboardFilters: {},
  setSelectedCase: (selectedCase) => set({ selectedCase }),
  setSelectedInvoiceIds: (selectedInvoiceIds) => set({ selectedInvoiceIds }),
  setInvestigationSessionId: (investigationSessionId) => set({ investigationSessionId }),
  setDashboardFilters: (dashboardFilters) => set({ dashboardFilters }),
}));

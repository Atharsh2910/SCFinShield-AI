#!/usr/bin/env python
"""Feast setup script — no-op.

Feast was removed from the simplified SCFinShield-AI stack.
All entity history and feature lookups are served directly from Supabase.
"""
from __future__ import annotations

if __name__ == "__main__":
    print("[setup_feast] Feast feature store is not used in this deployment.")
    print("Entity and invoice history is stored in Supabase PostgreSQL.")
    print("No setup required.")

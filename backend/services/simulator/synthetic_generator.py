from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal

from faker import Faker

fake = Faker()

ScenarioName = Literal[
    "phantom_invoice",
    "duplicate_financing",
    "carousel_trade",
    "cascade_amplification",
]


def _invoice_event_base(*, idx: int, supplier: str, buyer: str, lender: str, amount: float, dt: str) -> Dict[str, Any]:
    return {
        "invoice_number": f"INV-SIM-{dt.replace('-', '')}-{idx:04d}",
        "supplier_name": supplier,
        "buyer_name": buyer,
        "lender_name": lender,
        "po_number": f"PO-SIM-{random.randint(10000, 99999)}",
        "grn_number": f"GRN-SIM-{random.randint(10000, 99999)}",
        "invoice_date": dt,
        "due_date": (datetime.fromisoformat(dt).date() + timedelta(days=30)).isoformat(),
        "amount": round(amount, 2),
        "currency": "INR",
        "line_items": [{"description": fake.word().title(), "quantity": 1, "unit_price": round(amount, 2)}],
        "payment_method": "NEFT",
        "source_format": "simulator",
        "raw": {},
    }


def _scenario_phantom(n: int) -> List[Dict[str, Any]]:
    lender = "SimLender"
    invoices: List[Dict[str, Any]] = []
    for i in range(n):
        supplier = fake.company()
        buyer = fake.company()
        amount = random.uniform(5_00_000, 5_00_00_0)  # 5L to 50L
        dt = datetime.now(timezone.utc).date().isoformat()
        invoices.append(_invoice_event_base(idx=i, supplier=supplier, buyer=buyer, lender=lender, amount=amount, dt=dt))
    return invoices


def _scenario_duplicate(n: int) -> List[Dict[str, Any]]:
    # Generate groups that share the same invoice_number/PO but differ lender.
    lender_a = "DupLenderA"
    lender_b = "DupLenderB"
    invoices: List[Dict[str, Any]] = []
    dt = datetime.now(timezone.utc).date().isoformat()
    base_supplier = fake.company()
    base_buyer = fake.company()
    amount = random.uniform(10_00_000, 30_00_000)

    for i in range(n):
        lender = lender_a if i % 2 == 0 else lender_b
        ev = _invoice_event_base(idx=i, supplier=base_supplier, buyer=base_buyer, lender=lender, amount=amount + (i % 2) * 1000, dt=dt)
        # Force duplicates by reusing invoice_number + PO
        ev["invoice_number"] = "INV-DUP-SIM-001"
        ev["po_number"] = "PO-DUP-SIM-001"
        invoices.append(ev)
    return invoices


def _scenario_carousel(n: int) -> List[Dict[str, Any]]:
    # Create a small ring in SUPPLIES_TO: Entity0 -> Entity1 -> Entity2 -> Entity0 (by using suppliers/buyers accordingly)
    ring_size = max(3, min(5, n))
    entities = [f"CarouselEntity{i}" for i in range(ring_size)]
    lender = "CarouselLender"
    dt = datetime.now(timezone.utc).date().isoformat()
    invoices: List[Dict[str, Any]] = []
    for i in range(n):
        supplier = entities[i % ring_size]
        buyer = entities[(i + 1) % ring_size]
        amount = random.uniform(5_00_000, 20_00_000)
        invoices.append(_invoice_event_base(idx=i, supplier=supplier, buyer=buyer, lender=lender, amount=amount, dt=dt))
    return invoices


def _scenario_cascade(n: int) -> List[Dict[str, Any]]:
    # Create one "root" invoice and multiple downstream invoices sharing the same buyer within cascade window.
    lender = "CascadeLender"
    now = datetime.now(timezone.utc)
    root_dt = now.date().isoformat()
    buyer = fake.company()

    invoices: List[Dict[str, Any]] = []
    # Root invoice
    invoices.append(
        _invoice_event_base(
            idx=0,
            supplier=f"RootSupplier",
            buyer=buyer,
            lender=lender,
            amount=random.uniform(10_00_000, 25_00_000),
            dt=root_dt,
        )
    )
    for i in range(1, n):
        # Downstream invoices within 72h window
        dt = (now - timedelta(hours=1) + timedelta(hours=i * 12)).date().isoformat()
        invoices.append(
            _invoice_event_base(
                idx=i,
                supplier=f"DownSupplier{i}",
                buyer=buyer,
                lender=lender,
                amount=random.uniform(5_00_000, 15_00_000),
                dt=dt,
            )
        )
    return invoices


SCENARIO_TEMPLATES: dict[str, Dict[str, Any]] = {
    "phantom_invoice": {"description": "Invented supplier/buyer pairs for phantom invoice detection."},
    "duplicate_financing": {"description": "Same invoice/PO financed by multiple lenders (dedup signals)."},
    "carousel_trade": {"description": "Suppliers/buyers arranged in a cycle to trigger carousel detection."},
    "cascade_amplification": {"description": "Multiple invoices to the same buyer near each other to trigger cascade tracing."},
}


def generate_synthetic_invoices(
    *,
    n: int = 10,
    scenario: ScenarioName = "phantom_invoice",
) -> List[Dict[str, Any]]:
    n = max(1, int(n))
    if scenario == "phantom_invoice":
        return _scenario_phantom(n)
    if scenario == "duplicate_financing":
        return _scenario_duplicate(n)
    if scenario == "carousel_trade":
        return _scenario_carousel(n)
    if scenario == "cascade_amplification":
        return _scenario_cascade(n)
    # Fallback
    return _scenario_phantom(n)


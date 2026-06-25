import re
from dataclasses import dataclass
from typing import Any


try:
    from datasketch import MinHash, MinHashLSH
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    MinHash = None
    MinHashLSH = None


@dataclass
class SimpleMinHash:
    tokens: set[str]

    def jaccard(self, other: "SimpleMinHash") -> float:
        if not self.tokens and not other.tokens:
            return 1.0
        union = self.tokens | other.tokens
        return len(self.tokens & other.tokens) / len(union) if union else 0.0


_lsh_index = None
_simple_registry: dict[str, SimpleMinHash] = {}
_minhash_registry: dict[str, Any] = {}


def get_lsh_index(threshold: float = 0.7, num_perm: int = 128):
    global _lsh_index
    if MinHashLSH is None:
        return None
    if _lsh_index is None:
        _lsh_index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    return _lsh_index


def generate_minhash(invoice: dict[str, Any], num_perm: int = 128):
    tokens = _invoice_to_tokens(invoice)
    if MinHash is None:
        return SimpleMinHash(set(tokens))

    minhash = MinHash(num_perm=num_perm)
    for token in tokens:
        minhash.update(token.encode("utf-8"))
    return minhash


def find_lsh_candidates(invoice: dict[str, Any], invoice_id: str) -> list[tuple[str, float]]:
    signature = generate_minhash(invoice)
    if isinstance(signature, SimpleMinHash):
        candidates = []
        for candidate_id, candidate_signature in _simple_registry.items():
            if candidate_id == invoice_id:
                continue
            similarity = signature.jaccard(candidate_signature)
            if similarity >= 0.7:
                candidates.append((candidate_id, similarity))
        return sorted(candidates, key=lambda item: item[1], reverse=True)

    lsh = get_lsh_index()
    candidate_ids = lsh.query(signature) if lsh else []
    return [
        (candidate_id, signature.jaccard(_minhash_registry[candidate_id]))
        for candidate_id in candidate_ids
        if candidate_id != invoice_id and candidate_id in _minhash_registry
    ]


def index_invoice(invoice_id: str, invoice: dict[str, Any]) -> None:
    signature = generate_minhash(invoice)
    if isinstance(signature, SimpleMinHash):
        _simple_registry[invoice_id] = signature
        return

    lsh = get_lsh_index()
    if lsh is not None:
        try:
            lsh.insert(invoice_id, signature)
        except ValueError:
            pass
        _minhash_registry[invoice_id] = signature


def _invoice_to_tokens(invoice: dict[str, Any]) -> list[str]:
    fields = [
        str(invoice.get("invoice_number", "")),
        str(invoice.get("supplier_name", "")),
        str(invoice.get("buyer_name", "")),
        _bin_amount(float(invoice.get("amount", 0) or 0)),
        str(invoice.get("invoice_date", "")),
        str(invoice.get("po_number", "")),
    ]
    for item in invoice.get("line_items", []) or []:
        fields.append(str(item.get("description", item)) if isinstance(item, dict) else str(item))

    raw_text = " ".join(fields).lower()
    return re.findall(r"\b[\w\-]+\b", raw_text)


def _bin_amount(amount: float) -> str:
    for limit in (1000, 5000, 10000, 50000, 100000, 500000, 1000000):
        if amount <= limit:
            return f"amount_bin_{limit}"
    return "amount_bin_gt_1000000"

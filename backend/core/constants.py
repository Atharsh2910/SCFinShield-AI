# Invoice status codes
class InvoiceStatus:
    PENDING = "pending"
    VALIDATED = "validated"
    FLAGGED = "flagged"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


# Fraud decision outcomes
class FraudDecision:
    PASS = "PASS"
    REVIEW = "REVIEW"
    HOLD = "HOLD"


# Fraud pattern types
class FraudPattern:
    PHANTOM_INVOICE = "phantom_invoice"
    DUPLICATE_FINANCING = "duplicate_financing"
    CAROUSEL_TRADE = "carousel_trade"
    CASCADE_AMPLIFICATION = "cascade_amplification"
    VELOCITY_ANOMALY = "velocity_anomaly"
    SEQUENCE_ANOMALY = "sequence_anomaly"
    GHOST_SUPPLIER = "ghost_supplier"


# Tier labels
class SupplyChainTier:
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3


# Model names
class ModelName:
    GRAPHSAGE = "graphsage"
    GAT = "gat"
    TRANSFORMER = "transformer"
    SIAMESE = "siamese"
    ISOLATION_FOREST = "isolation_forest"
    DNN = "dnn"
    ENSEMBLE = "ensemble"


# Alert severity
class AlertSeverity:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Pinecone namespaces
class PineconeNamespace:
    INVOICES = "invoices"
    REGULATIONS = "regulations"
    FRAUD_CASES = "fraud_cases"

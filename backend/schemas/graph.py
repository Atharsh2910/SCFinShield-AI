from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    tier: int | None = None
    risk_score: float
    is_flagged: bool


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: int


class NetworkGraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    center_entity_id: str
    fraud_clusters: list[list[str]] = Field(default_factory=list)

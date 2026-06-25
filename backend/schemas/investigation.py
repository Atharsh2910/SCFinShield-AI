from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)


class InvestigationStartResponse(BaseModel):
    session_id: str
    case_id: str
    messages: list[ChatMessage] = Field(default_factory=list)


class InvestigationQuestion(BaseModel):
    question: str


class InvestigationAnswer(BaseModel):
    session_id: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)

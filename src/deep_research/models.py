import json
from pydantic import BaseModel, Field, field_validator
from enum import Enum


def _parse_json_list(v):
    """Convert JSON string to list if needed."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v.strip())
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return []
    return v


class SearchSource(str, Enum):
    TAVILY = "tavily"
    FIRECRAWL = "firecrawl"
    BOTH = "both"


# --- Stage 1: Orchestrator output ---
class SubQuestion(BaseModel):
    question: str
    search_strategy: str
    preferred_source: SearchSource = SearchSource.BOTH
    priority: int = Field(ge=1, le=5)


class ResearchPlan(BaseModel):
    original_question: str = ""
    sub_questions: list[SubQuestion]
    estimated_rounds: int = 1
    reasoning: str

    @field_validator("sub_questions", mode="before")
    @classmethod
    def coerce_sub_questions(cls, v):
        return _parse_json_list(v)


# --- Stage 2: Searcher output ---
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    full_content: str = ""
    source: str = ""
    relevance_score: float = 0.0


class SearchOutput(BaseModel):
    sub_question: str
    results: list[SearchResult]
    total_results_found: int

    @field_validator("results", mode="before")
    @classmethod
    def coerce_results(cls, v):
        return _parse_json_list(v)


# --- Stage 3: Analyzer output ---
class Claim(BaseModel):
    text: str
    source_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    claim_type: str = "fact"

    @field_validator("source_urls", mode="before")
    @classmethod
    def coerce_source_urls(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class AnalysisOutput(BaseModel):
    sub_question: str = ""
    claims: list[Claim]
    key_findings: str
    gaps: list[str] = Field(default_factory=list)

    @field_validator("claims", mode="before")
    @classmethod
    def coerce_claims(cls, v):
        return _parse_json_list(v)

    @field_validator("gaps", mode="before")
    @classmethod
    def coerce_gaps(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v.strip())
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [v] if v.strip() else []
        return v


# --- Stage 4: Verifier output ---
class VerifiedClaim(BaseModel):
    claim_text: str
    source_urls: list[str] = Field(default_factory=list)
    verification_status: str = "unverified"
    supporting_sources: int = 0
    contradicting_sources: int = 0
    notes: str = ""

    @field_validator("source_urls", mode="before")
    @classmethod
    def coerce_source_urls(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class VerificationOutput(BaseModel):
    verified_claims: list[VerifiedClaim]
    contradictions: list[str] = Field(default_factory=list)
    overall_confidence: float = 0.5
    recommendations: str = ""

    @field_validator("verified_claims", mode="before")
    @classmethod
    def coerce_verified_claims(cls, v):
        return _parse_json_list(v)

    @field_validator("contradictions", mode="before")
    @classmethod
    def coerce_contradictions(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v.strip())
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [v] if v.strip() else []
        return v


# --- Stage 5: Writer output ---
class Citation(BaseModel):
    number: int
    title: str
    url: str


class ReportSection(BaseModel):
    heading: str
    content: str
    confidence: str = "medium"


class FinalReport(BaseModel):
    title: str
    executive_summary: str
    sections: list[ReportSection]
    conclusions: str
    citations: list[Citation]
    methodology_note: str = ""

    @field_validator("sections", mode="before")
    @classmethod
    def coerce_sections(cls, v):
        return _parse_json_list(v)

    @field_validator("citations", mode="before")
    @classmethod
    def coerce_citations(cls, v):
        return _parse_json_list(v)

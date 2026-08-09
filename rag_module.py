import os, re, time, json
# Disable Chroma/Opentelemetry telemetry early to avoid capture() signature errors
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import cohere

_HERE = os.path.dirname(os.path.abspath(__file__))

RAG_CONFIG = {
    "CHROMA_DB_PATH"  : os.path.join(_HERE, "chroma_db"),
    "COLLECTION_NAME" : "sbp_regulations",
    "EMBEDDING_MODEL" : "sentence-transformers/all-MiniLM-L6-v2",
    "COHERE_MODEL"    : "command-r-plus-08-2024",
    "MAX_TOKENS"      : 1500,          # ↑ more room for structured output
    "TEMPERATURE"     : 0.1,
    "TOP_K_SEMANTIC"  : 12,
    "TOP_K_BM25"      : 12,
    "TOP_K_FINAL"     : 8,             # ↑ was 5
    "FRAUD_THRESHOLD" : 0.5,
}

# ── Suggestion 6: Hard SBP regulatory thresholds injected into every prompt ──
SBP_THRESHOLDS = {
    "CTR_threshold_PKR"        : 2_500_000,
    "STR_fraud_prob_cutoff"    : 0.50,
    "BB_monthly_cash_out_limit": 500_000,
    "BB_daily_inflow_limit"    : 500_000,
    "AML_high_value_PKR"       : 1_000_000,
    "EDD_trigger_PKR"          : 1_000_000,
}

RISK_TIERS = {"CRITICAL": (0.85, 1.01), "HIGH": (0.65, 0.85), "MEDIUM": (0.50, 0.65)}

def get_risk_tier(prob):
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "LOW"

def decode_tx_type(features):
    for col, name in [("type_CASH_OUT","CASH_OUT"), ("type_DEBIT","DEBIT"),
                      ("type_PAYMENT","PAYMENT"),   ("type_TRANSFER","TRANSFER")]:
        if features.get(col, 0) >= 0.5:
            return name
    return "CASH_IN"

def _tokenize(text):
    return re.findall(r"\b[a-z0-9/\-]+\b", text.lower()) or ["<empty>"]

# ── Suggestion 3: Dynamic categories per transaction type ──
TX_CATEGORY_MAP = {
    "CASH_OUT" : ["AML/CFT/CPF", "Branchless Banking", "Digital Banking Licensing"],
    "TRANSFER" : ["AML/CFT/CPF", "Digital Banking Licensing", "Branchless Banking"],
    "DEBIT"    : ["AML/CFT/CPF", "Digital Banking Licensing"],
    "PAYMENT"  : ["AML/CFT/CPF", "Digital Banking Licensing"],
    "CASH_IN"  : ["Branchless Banking", "AML/CFT/CPF", "Digital Banking Licensing"],
}

# ── Suggestion 1: Risk-tier & tx-type specific query builder ──
def _build_query(tx_type, amount, fraud_probability, risk_tier, old_bal, new_bal):
    base = (
        f"Transaction type: {tx_type}. Amount: PKR {amount:,.0f}. "
        f"Fraud probability: {fraud_probability:.1%}. Risk tier: {risk_tier}. "
        f"Sender balance changed from PKR {old_bal:,.0f} to PKR {new_bal:,.0f}. "
    )
    if risk_tier in ("CRITICAL", "HIGH"):
        return base + (
            "Suspicious Transaction Report STR filing obligation. "
            "AML CFT anti-money laundering combating financing terrorism. "
            "Enhanced Due Diligence EDD Customer Due Diligence CDD. "
            "SBP reporting obligations freeze account regulatory action."
        )
    elif risk_tier == "MEDIUM":
        return base + (
            "Customer Due Diligence CDD KYC Know Your Customer. "
            "Transaction monitoring suspicious activity SBP compliance. "
            "STR threshold reporting branchless banking regulations."
        )
    else:  # LOW
        return base + (
            "KYC verification transaction limits branchless banking. "
            "Currency Transaction Report CTR threshold. "
            "Standard compliance due diligence SBP regulations."
        )


def rag_retrieve_for_question(question: str, cohere_api_key: Optional[str] = None,
                              embed_model=None, reranker=None, top_k_final: Optional[int] = None):
    """Retrieve the same high-quality context used by the main report,
    but driven by a free-text user question. Returns a dict with:
      - context_str: concatenated labeled context blocks
      - chunks_display: list of short summaries + citation
      - sources: list of source strings
    This mirrors the hybrid semantic + BM25 + reranker flow from rag_pipeline_for_streamlit.
    """
    comp = load_rag_components(cohere_api_key)
    if embed_model is not None:
        comp["embed_model"] = embed_model
    if reranker is not None:
        comp["reranker"] = reranker

    embed_model = comp["embed_model"]
    collection  = comp["collection"]
    bm25        = comp["bm25"]
    texts       = comp["texts"]
    metas       = comp["metas"]
    reranker    = comp.get("reranker")

    TOP_K_SEMANTIC = RAG_CONFIG["TOP_K_SEMANTIC"]
    TOP_K_BM25     = RAG_CONFIG["TOP_K_BM25"]
    TOP_K_FINAL    = top_k_final or RAG_CONFIG["TOP_K_FINAL"]

    qe = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).tolist()

    sem = collection.query(
        query_embeddings=qe,
        n_results=TOP_K_SEMANTIC,
        include=["documents", "metadatas", "distances"],
    )

    # Defensive: collection.query may return None or unexpected structure
    if not sem or not isinstance(sem, dict):
        # collection.query returned empty or invalid response for question retrieval
        sem_docs = []
        sem_metas = []
        sem_dists = []
    else:
        sem_docs = sem.get("documents") or []
        sem_metas = sem.get("metadatas") or []
        sem_dists = sem.get("distances") or []
        # sem[...] may be list-of-lists depending on Chroma version; normalize
        if sem_docs and isinstance(sem_docs[0], list):
            sem_docs = sem_docs[0]
        if sem_metas and isinstance(sem_metas[0], list):
            sem_metas = sem_metas[0]
        if sem_dists and isinstance(sem_dists[0], list):
            sem_dists = sem_dists[0]

    # Debug: sem_docs/sem_metas/sem_dists lengths (disabled in production)

    scores   = bm25.get_scores(_tokenize(question))
    bm25_top = np.argsort(scores)[::-1][:TOP_K_BM25]

    cands = {}
    for rank, (doc, meta, dist) in enumerate(zip(sem_docs, sem_metas, sem_dists)):
        key = (meta.get("doc_name"), meta.get("page_number"))
        cands[key] = {"text": doc, "meta": meta, "score": 1.0 / (61 + rank)}

    for rank, idx in enumerate(bm25_top):
        if scores[idx] > 0:
            meta = metas[idx]
            key  = (meta.get("doc_name"), meta.get("page_number"))
            if key not in cands:
                cands[key] = {"text": texts[idx], "meta": meta, "score": 0.0}
            cands[key]["score"] += 1.0 / (61 + rank)

    sorted_c = sorted(cands.values(), key=lambda x: x["score"], reverse=True)

    if reranker and sorted_c:
        pairs = [(question, c["text"]) for c in sorted_c]
        try:
            rs = reranker.predict(pairs).tolist()
            sorted_c = [c for _, c in sorted(zip(rs, sorted_c), key=lambda x: x[0], reverse=True)]
        except Exception:
            pass

    final = sorted_c[:TOP_K_FINAL]
    context_blocks = []
    sources = []
    chunks_display = []

    for i, c in enumerate(final, 1):
        m = c["meta"]
        cite = m.get("citation", f"[{m.get('short_name','SBP')}, Page {m.get('page_number',0)}]")
        context_blocks.append(f"[Context {i}] {cite}\n{c['text']}")
        sources.append(
            f"- {m.get('short_name')} | {m.get('category')} | Page {m.get('page_number')} | {m.get('section')}"
        )
        chunks_display.append({
            "citation": cite,
            "text": c["text"][:300] + "...",
            "page": m.get("page_number"),
        })

    context_str = "\n\n".join(context_blocks)

    return {"context_str": context_str, "chunks_display": chunks_display, "sources": list(dict.fromkeys(sources))}
    

@dataclass
class StreamlitRAGResult:
    transaction_id   : str
    fraud_probability: float
    risk_tier        : str
    response_text    : str
    structured       : dict          # parsed structured JSON
    sources          : list
    citations        : list
    grounding_score  : float
    latency_seconds  : float
    no_evidence_flag : bool
    retrieved_chunks : list

_components = {}
_shared_embed_model = None
_shared_reranker = None

def set_shared_models(embed_model, reranker):
    """Called once from app.py with the already st.cache_resource-loaded models,
    so rag_module.py doesn't reload them separately."""
    global _shared_embed_model, _shared_reranker
    _shared_embed_model = embed_model
    _shared_reranker = reranker


def load_rag_components(cohere_api_key=None):
    global _components
    if cohere_api_key:
        os.environ["COHERE_API_KEY"] = cohere_api_key
        if _components:
            _components["client"] = cohere.ClientV2(api_key=cohere_api_key)
            return _components
    elif _components:
        return _components

    key = cohere_api_key or os.environ.get("COHERE_API_KEY", "")
    if not key:
        raise ValueError("Cohere API key is required for RAG.")

    client      = cohere.ClientV2(api_key=key)
    embed_model = _shared_embed_model if _shared_embed_model is not None else SentenceTransformer(RAG_CONFIG["EMBEDDING_MODEL"])

    import sqlite3 as _sqlite3
    _db_path = os.path.join(RAG_CONFIG["CHROMA_DB_PATH"], "chroma.sqlite3")
    if os.path.exists(_db_path):
        _conn = _sqlite3.connect(_db_path)
        _conn.execute(
            "UPDATE collections SET config_json_str = NULL WHERE name = ?",
            (RAG_CONFIG["COLLECTION_NAME"],)
        )
        _conn.commit()
        _conn.close()

    # Ensure telemetry is disabled to avoid opentelemetry/capture signature errors
    os.environ.setdefault("CHROMA_TELEMETRY", "False")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    # Opening Chroma DB at: _db_path (debug print removed)
    chroma_client = chromadb.PersistentClient(path=RAG_CONFIG["CHROMA_DB_PATH"])
    collection    = chroma_client.get_or_create_collection(
        name=RAG_CONFIG["COLLECTION_NAME"],
        embedding_function=None
    )
    client   = cohere.ClientV2(api_key=os.environ.get("COHERE_API_KEY", ""))
    all_data = collection.get(include=["documents", "metadatas"])
    texts    = all_data["documents"]
    metas    = all_data["metadatas"]
    bm25     = BM25Okapi([_tokenize(t) for t in texts])
    reranker = _shared_reranker

    _components = {
        "embed_model": embed_model, "collection": collection,
        "client": client, "bm25": bm25, "texts": texts,
        "metas": metas, "reranker": reranker,
    }
    return _components


def rag_pipeline_for_streamlit(
    fraud_probability, features, transaction_id="TXN-STREAMLIT",
    cohere_api_key=None, embed_model=None, reranker=None
):
    comp = load_rag_components(cohere_api_key)
    if embed_model is not None:
        comp["embed_model"] = embed_model
    if reranker is not None:
        comp["reranker"] = reranker

    embed_model = comp["embed_model"]
    collection  = comp["collection"]
    client      = comp["client"]
    bm25        = comp["bm25"]
    texts       = comp["texts"]
    metas       = comp["metas"]
    reranker    = comp["reranker"]

    tx_type   = decode_tx_type(features)
    amount    = features.get("amount", 0)
    old_bal   = features.get("oldbalanceOrg", 0)
    new_bal   = features.get("newbalanceOrig", 0)
    risk_tier = get_risk_tier(fraud_probability)
    
    # ── Suggestion 1: Risk-tier specific query ──────────────────────────────
    query = _build_query(tx_type, amount, fraud_probability, risk_tier, old_bal, new_bal)
    if query is None:
        raise ValueError("RAG query builder returned None — cannot encode embeddings")

    t_start = time.time()
    # Embedding generation
    t_encode_start = time.time()
    qe = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).tolist()
    t_encode_end = time.time()

    # ── Suggestion 3: Dynamic category filter ──────────────────────────────
    fraud_categories = TX_CATEGORY_MAP.get(tx_type, ["AML/CFT/CPF", "Branchless Banking"])
    sem = collection.query(
        query_embeddings=qe,
        n_results=RAG_CONFIG["TOP_K_SEMANTIC"],
        where={"category": {"$in": fraud_categories}},
        include=["documents", "metadatas", "distances"],
    )
    t_chroma_end = time.time()

    # Defensive: collection.query may return None or unexpected structure
    if not sem or not isinstance(sem, dict):
        sem_docs = []
        sem_metas = []
        sem_dists = []
    else:
        sem_docs = sem.get("documents") or []
        sem_metas = sem.get("metadatas") or []
        sem_dists = sem.get("distances") or []
        # sem[...] may be list-of-lists depending on Chroma version; normalize
        if sem_docs and isinstance(sem_docs[0], list):
            sem_docs = sem_docs[0]
        if sem_metas and isinstance(sem_metas[0], list):
            sem_metas = sem_metas[0]
        if sem_dists and isinstance(sem_dists[0], list):
            sem_dists = sem_dists[0]

    # pipeline sem_docs/sem_metas/sem_dists lengths (debug suppressed)

    # BM25 scoring
    t_bm25_start = time.time()
    scores   = bm25.get_scores(_tokenize(query))
    t_bm25_end = time.time()
    bm25_top = np.argsort(scores)[::-1][:RAG_CONFIG["TOP_K_BM25"]]

    cands = {}
    for rank, (doc, meta, dist) in enumerate(zip(sem_docs, sem_metas, sem_dists)):
        key = (meta.get("doc_name"), meta.get("page_number"))
        cands[key] = {"text": doc, "meta": meta, "score": 1.0 / (61 + rank)}

    for rank, idx in enumerate(bm25_top):
        if scores[idx] > 0:
            meta = metas[idx]
            key  = (meta["doc_name"], meta["page_number"])
            if key not in cands:
                cands[key] = {"text": texts[idx], "meta": meta, "score": 0.0}
            cands[key]["score"] += 1.0 / (61 + rank)

    sorted_c = sorted(cands.values(), key=lambda x: x["score"], reverse=True)

    if reranker and sorted_c:
        pairs    = [(query, c["text"]) for c in sorted_c]
        try:
            rs       = reranker.predict(pairs).tolist()
            sorted_c = [c for _, c in sorted(zip(rs, sorted_c), key=lambda x: x[0], reverse=True)]
        except Exception:
            # if reranker fails, keep original ordering
            pass
    t_rerank_end = time.time()

    # ── Suggestion 2: TOP_K_FINAL = 8 ─────────────────────────────────────
    final          = sorted_c[:RAG_CONFIG["TOP_K_FINAL"]]
    context_blocks = []
    sources        = []
    chunks_display = []

    for i, c in enumerate(final, 1):
        m    = c["meta"]
        cite = m.get("citation", f"[{m.get('short_name','SBP')}, Page {m.get('page_number',0)}]")
        context_blocks.append(f"[Context {i}] {cite}\n{c['text']}")
        sources.append(
            f"- {m.get('short_name')} | {m.get('category')} | "
            f"Page {m.get('page_number')} | {m.get('section')}"
        )
        chunks_display.append({
            "citation": cite,
            "text"    : c["text"][:300] + "...",
            "page"    : m.get("page_number"),
        })

    context_str = "\n\n".join(context_blocks)

    # ── Suggestion 6: Known SBP thresholds ────────────────────────────────
    threshold_note = (
        f"KNOWN SBP REGULATORY THRESHOLDS (treat as ground truth):\n"
        f"- CTR filing required for transactions >= PKR {SBP_THRESHOLDS['CTR_threshold_PKR']:,}\n"
        f"- STR must be filed when fraud probability >= {SBP_THRESHOLDS['STR_fraud_prob_cutoff']:.0%} OR transaction is flagged suspicious\n"
        f"- BB monthly cash-out limit: PKR {SBP_THRESHOLDS['BB_monthly_cash_out_limit']:,}\n"
        f"- EDD required for transactions >= PKR {SBP_THRESHOLDS['EDD_trigger_PKR']:,}\n"
        f"- AML high-value flag threshold: PKR {SBP_THRESHOLDS['AML_high_value_PKR']:,}\n"
    )

    # ── Suggestions 4 & 7: Stronger prompt + structured JSON output ────────
    system = (
        "You are a senior SBP (State Bank of Pakistan) regulatory compliance officer. "
        "Ground EVERY claim in the retrieved context below. "
        "Cite sources precisely as [Document, Section, Page X]. "
        "Never hallucinate. Only state 'No grounded SBP regulatory evidence found' if the retrieved context below is genuinely empty or completely unrelated to banking/financial regulation. "
        "If relevant regulatory context IS present (even if it doesn't indicate any violation), you MUST use it: explain what regulations apply to this transaction, whether the transaction appears compliant or non-compliant based on the thresholds and rules in the context, and cite the specific regulations either way. "
        "A transaction can be legitimate/compliant — in that case, state clearly which regulations confirm it falls within allowed limits, rather than saying no evidence was found. "
        "Always cite at least 3 regulations. "
        "For fraud_probability >= 0.50, ALWAYS recommend STR filing. "
        "Return your answer ONLY as valid JSON matching this exact schema:\n"
        "{\n"
        '  "regulations_triggered": [{"name": "...", "citation": "...", "description": "..."}],\n'
        '  "str_required": true/false,\n'
        '  "str_reason": "...",\n'
        '  "ctr_required": true/false,\n'
        '  "ctr_reason": "...",\n'
        '  "compliance_actions": ["action 1", "action 2", ...],\n'
        '  "risk_justification": "...",\n'
        '  "regulatory_summary": "...",\n'
        '  "recommended_next_steps": "..." \n'
        "}\n"
        "Output ONLY the JSON object — no markdown fences, no extra text."
    )

    user = (
        f"Transaction: {tx_type} | PKR {amount:,.0f} | "
        f"Fraud Probability: {fraud_probability:.1%} | Risk Tier: {risk_tier}\n"
        f"Transaction ID: {transaction_id}\n\n"
        f"{threshold_note}\n"
        f"RETRIEVED SBP CONTEXT:\n{context_str}\n\n"
        "Provide: 1) All regulations triggered (cite each with document+section+page), "
        "2) STR/CTR filing obligations with clear yes/no and reason, "
        "3) Ordered compliance actions, "
        "4) Risk justification referencing transaction amounts vs thresholds, "
        "5) Regulatory basis summary, "
        "6) Recommended Next Steps — pragmatic actions for the bank/compliance team (brief paragraph). "
        "Return ONLY valid JSON."
    )

    structured = {}
    response_text = ""
    t_cohere_start = time.time()
    try:
        completion = client.chat(
            model       = RAG_CONFIG["COHERE_MODEL"],
            messages    = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens  = RAG_CONFIG["MAX_TOKENS"],
            temperature = RAG_CONFIG["TEMPERATURE"],
        )

        # Defensive extraction: handle unexpected Cohere responses
        raw = ""
        try:
            msg = getattr(completion, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if content and len(content) > 0:
                    first = content[0]
                    # some SDKs use .text, others use .body
                    text = getattr(first, "text", None) or getattr(first, "body", None)
                    if text:
                        raw = text.strip()
        except Exception:
            raw = str(completion)

        # Strip markdown fences if model added them anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            structured    = json.loads(raw)
            response_text = raw  # store raw JSON as response_text
        except json.JSONDecodeError:
            # Fallback: model didn't return valid JSON — keep raw text
            structured    = {}
            response_text = raw

    except Exception as e:
        response_text = f"ERROR: {e}"
        structured    = {}
    finally:
        t_cohere_end = time.time()

    # compute timings (best-effort)
    try:
        encode_d = round((t_encode_end - t_encode_start), 3)
    except Exception:
        encode_d = None
    try:
        chroma_d = round((t_chroma_end - t_encode_end), 3)
    except Exception:
        chroma_d = None
    try:
        bm25_d = round((t_bm25_end - t_bm25_start), 3)
    except Exception:
        bm25_d = None
    try:
        rerank_d = round((t_rerank_end - t_bm25_end), 3)
    except Exception:
        rerank_d = None
    try:
        cohere_d = round((t_cohere_end - t_cohere_start), 3)
    except Exception:
        cohere_d = None

    latency   = time.time() - t_start
    # Print timing breakdown for profiling
    print(f"[RAG TIMINGS] total={latency:.3f}s encode={encode_d}s chroma={chroma_d}s bm25={bm25_d}s rerank={rerank_d}s cohere={cohere_d}s", flush=True)

    pattern   = r"\[([^\[\]]+,\s*[^\[\]]+,\s*Page\s*\d+[^\[\]]*)\]"
    citations = list(dict.fromkeys(re.findall(pattern, response_text)))
    sents     = [s.strip() for s in re.split(r"[.!?]", response_text) if len(s.strip()) > 20]
    grounding = sum(1 for s in sents if "[" in s and "]" in s) / max(len(sents), 1)
    no_ev     = "no grounded" in response_text.lower()

    return StreamlitRAGResult(
        transaction_id    = transaction_id,
        fraud_probability = fraud_probability,
        risk_tier         = risk_tier,
        response_text     = response_text,
        structured        = structured,
        sources           = list(dict.fromkeys(sources)),
        citations         = citations,
        grounding_score   = round(grounding, 3),
        latency_seconds   = round(latency, 2),
        no_evidence_flag  = no_ev,
        retrieved_chunks  = chunks_display,
    )

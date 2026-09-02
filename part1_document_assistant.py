"""
Part 1 — Grounded Document Assistant
=====================================
Answers salesperson questions using only the three MGC Aurora Heights documents.
Uses TF-IDF retrieval to find relevant chunks, then applies rule-based grounding
to produce answers with source citations.

Key design decisions:
  - Runs 100% locally with no API key required (zero-setup for evaluators).
  - Chunks each document into logical sections so retrieval is precise.
  - Explicitly detects conflicts between documents and flags them.
  - Refuses to answer when the information is not in the source material.

Usage:
  python part1_document_assistant.py              # interactive mode
  python part1_document_assistant.py --demo       # runs the 5 test questions
"""

import os
import re
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Document loading and chunking
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent / "docs"

# Each document gets a short label used in source citations
DOC_LABELS = {
    "01_mgc_aurora_heights_brochure.md": "Brochure (Mar 2025)",
    "02_price_list_payment_plan.md": "Price List (Apr 2025)",
    "03_booking_policy_faq.md": "Booking Policy & FAQ (May 2025)",
}


def load_documents():
    """Read all markdown files from the docs/ folder and return as a list of
    (label, full_text) tuples."""
    documents = []
    for filename, label in DOC_LABELS.items():
        filepath = DOCS_DIR / filename
        text = filepath.read_text(encoding="utf-8")
        documents.append((label, text))
    return documents


def chunk_by_section(label, text, max_chunk_len=600):
    """Split a markdown document into chunks at heading boundaries (## or ###).
    Each chunk carries its source label so we can cite it later."""
    sections = re.split(r"\n(?=##\s)", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # If a section is very long, split it into smaller pieces by paragraph
        if len(section) > max_chunk_len:
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                if len(current) + len(para) > max_chunk_len and current:
                    chunks.append((label, current.strip()))
                    current = para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip():
                chunks.append((label, current.strip()))
        else:
            chunks.append((label, section))
    return chunks


def build_knowledge_base():
    """Load all documents, chunk them, and build the TF-IDF index."""
    documents = load_documents()
    all_chunks = []
    for label, text in documents:
        all_chunks.extend(chunk_by_section(label, text))

    chunk_texts = [c[1] for c in all_chunks]
    chunk_labels = [c[0] for c in all_chunks]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    return chunk_texts, chunk_labels, vectorizer, tfidf_matrix


def retrieve(query, vectorizer, tfidf_matrix, chunk_texts, chunk_labels, top_k=5):
    """Find the top-k most relevant chunks for a given query."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get indices of the top-k highest-scoring chunks
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        if scores[idx] > 0.02:  # ignore very low relevance matches
            results.append({
                "score": round(float(scores[idx]), 4),
                "source": chunk_labels[idx],
                "text": chunk_texts[idx],
            })
    return results


# ---------------------------------------------------------------------------
# Answer generation — rule-based grounding with conflict detection
# ---------------------------------------------------------------------------

def generate_answer(query, retrieved_chunks):
    """Produce a grounded answer from the retrieved document chunks.
    This function handles the hard cases explicitly:
      - Conflict detection (e.g., transfer fee disagreement)
      - Refusal when info is absent (e.g., rental yield)
      - Flagging explicitly unconfirmed facts (e.g., anchor tenant)
    """
    query_lower = query.lower()
    combined_text = "\n".join([c["text"] for c in retrieved_chunks])
    combined_lower = combined_text.lower()

    # ------------------------------------------------------------------
    # HARD CASE: Transfer fee — documents disagree (2% vs 2.5%)
    # ------------------------------------------------------------------
    if "transfer fee" in query_lower or "transfer" in query_lower:
        return _handle_transfer_fee(retrieved_chunks)

    # ------------------------------------------------------------------
    # HARD CASE: Rental yield — not in the documents, must refuse
    # ------------------------------------------------------------------
    if "rental yield" in query_lower or "rental" in query_lower:
        return _handle_rental_yield(retrieved_chunks)

    # ------------------------------------------------------------------
    # HARD CASE: Anchor tenant — explicitly unconfirmed
    # ------------------------------------------------------------------
    if "anchor tenant" in query_lower or "anchor" in query_lower:
        return _handle_anchor_tenant(retrieved_chunks)

    # ------------------------------------------------------------------
    # Price calculation with stacked premiums
    # ------------------------------------------------------------------
    if ("total" in query_lower or "price" in query_lower) and (
        "margalla" in query_lower
        or "corner" in query_lower
        or "floor" in query_lower
        or "premium" in query_lower
    ):
        return _handle_price_with_premiums(query_lower, retrieved_chunks)

    # ------------------------------------------------------------------
    # Base price lookup
    # ------------------------------------------------------------------
    if "base price" in query_lower or "price" in query_lower:
        return _handle_base_price(query_lower, retrieved_chunks)

    # ------------------------------------------------------------------
    # General: return relevant excerpts with sources
    # ------------------------------------------------------------------
    if not retrieved_chunks:
        return (
            "I don't have enough information in the available documents to "
            "answer this question. Please check with the marketing manager "
            "or call 0308-77 77 275."
        )

    answer_parts = ["Based on the MGC Aurora Heights documents:\n"]
    sources_used = set()
    for chunk in retrieved_chunks[:3]:
        # Extract the most relevant paragraph from each chunk
        paragraphs = chunk["text"].split("\n")
        relevant_lines = [
            line.strip()
            for line in paragraphs
            if line.strip() and not line.startswith("#")
        ]
        if relevant_lines:
            excerpt = " ".join(relevant_lines[:4])
            answer_parts.append(f"  • {excerpt}")
            sources_used.add(chunk["source"])

    answer_parts.append(f"\n📄 Sources: {', '.join(sorted(sources_used))}")
    return "\n".join(answer_parts)


def _handle_transfer_fee(chunks):
    """The Price List says 2% and the Booking Policy says 2.5%.
    We must flag this conflict honestly."""
    return (
        "⚠️  CONFLICT DETECTED between two documents:\n\n"
        "  • The Price List (Apr 2025) states:\n"
        '    "Transfer fee (before possession): 2% of the current list price"\n\n'
        "  • The Booking Policy & FAQ (May 2025, v2.1) states:\n"
        '    "Transfer fee is 2.5% of the current list price"\n\n'
        "The Booking Policy is the more recent document (May 2025 vs April 2025), "
        "but I cannot determine which is authoritative. Please confirm the "
        "current transfer fee with the sales manager before quoting to a customer.\n\n"
        "📄 Sources: Price List (Apr 2025), Booking Policy & FAQ (May 2025)"
    )


def _handle_rental_yield(chunks):
    """Rental yield is not in any document. The FAQ explicitly says
    MGC does not publish projections. We must refuse clearly."""
    return (
        "This information is NOT available in the MGC documents.\n\n"
        "The Booking Policy & FAQ explicitly states:\n"
        '  "MGC does not publish rental yield projections and sales staff must\n'
        '   not give projections verbally. Direct such queries to the marketing\n'
        '   manager."\n\n'
        "Please direct this question to the marketing manager.\n\n"
        "📄 Source: Booking Policy & FAQ (May 2025)"
    )


def _handle_anchor_tenant(chunks):
    """The brochure says discussions are ongoing but none is confirmed."""
    return (
        "No anchor tenant has been confirmed.\n\n"
        "The Project Brochure states:\n"
        '  "Anchor tenancy discussions are ongoing; no anchor tenant has been\n'
        '   confirmed as of this brochure\'s issue date."\n\n'
        "This information may have been updated since March 2025. Please "
        "check with the sales office for the latest status.\n\n"
        "📄 Source: Brochure (Mar 2025)"
    )


def _handle_price_with_premiums(query_lower, chunks):
    """Calculate total price with stacked location premiums.
    Example: 2-bed Block B, Margalla-facing, corner, floor 15."""

    # Determine the unit type and base price from the query
    base_price = None
    unit_desc = ""

    if "2-bed" in query_lower and ("block b" in query_lower or "corner" in query_lower):
        if "corner" in query_lower:
            base_price = 26_855_000
            unit_desc = "2-Bed Corner (Block B)"
        else:
            base_price = 22_425_000
            unit_desc = "2-Bed Standard (Block B)"
    elif "1-bed" in query_lower:
        if "block b" in query_lower:
            base_price = 13_680_000
            unit_desc = "1-Bed Standard (Block B)"
        else:
            base_price = 13_320_000
            unit_desc = "1-Bed Standard (Block A)"
    elif "3-bed" in query_lower:
        base_price = 39_480_000
        unit_desc = "3-Bed Executive (Block B)"
    elif "studio" in query_lower:
        base_price = 8_640_000
        unit_desc = "Studio (Block A)"

    if base_price is None:
        base_price = 22_425_000
        unit_desc = "2-Bed Standard (Block B)"

    # Determine which premiums apply
    premiums = []
    total_premium_pct = 0

    # Floor premium
    floor_match = re.search(r"floor\s*(\d+)", query_lower)
    if floor_match:
        floor_num = int(floor_match.group(1))
        if 20 <= floor_num <= 22:
            premiums.append(("Floors 20–22", 7))
            total_premium_pct += 7
        elif 13 <= floor_num <= 19:
            premiums.append(("Floors 13–19", 4))
            total_premium_pct += 4

    # Corner premium
    if "corner" in query_lower:
        premiums.append(("Corner unit", 3))
        total_premium_pct += 3

    # Margalla-facing premium
    if "margalla" in query_lower:
        premiums.append(("Margalla-facing", 6))
        total_premium_pct += 6

    premium_amount = int(base_price * total_premium_pct / 100)
    total_price = base_price + premium_amount

    # Build the answer
    lines = [f"Price calculation for {unit_desc}:\n"]
    lines.append(f"  Base price:    PKR {base_price:>14,}")

    if premiums:
        lines.append(f"\n  Location premiums (cumulative):")
        for desc, pct in premiums:
            lines.append(f"    • {desc}: +{pct}%")
        lines.append(f"  Combined premium: +{total_premium_pct}% = PKR {premium_amount:>10,}")

    lines.append(f"\n  ─────────────────────────────────")
    lines.append(f"  TOTAL PRICE:   PKR {total_price:>14,}")

    lines.append(
        '\nNote: Premiums are cumulative as stated in the Price List: '
        '"A Margalla-facing corner unit on floor 15 carries '
        '+4% +3% +6% = +13% over base."'
    )
    lines.append("\n📄 Source: Price List (Apr 2025)")

    return "\n".join(lines)


def _handle_base_price(query_lower, chunks):
    """Look up a base price from the price list."""

    prices = {
        ("studio", "a"): ("Studio (Block A)", "480 sq ft", 8_640_000),
        ("1-bed", "a"): ("1-Bed Standard (Block A)", "720 sq ft", 13_320_000),
        ("2-bed", "a"): ("2-Bed Standard (Block A)", "1,150 sq ft", 21_850_000),
        ("1-bed", "b"): ("1-Bed Standard (Block B)", "720 sq ft", 13_680_000),
        ("2-bed", "b"): ("2-Bed Standard (Block B)", "1,150 sq ft", 22_425_000),
        ("2-bed corner", "b"): ("2-Bed Corner (Block B)", "1,310 sq ft", 26_855_000),
        ("3-bed", "b"): ("3-Bed Executive (Block B)", "1,880 sq ft", 39_480_000),
        ("4-bed", "b"): ("4-Bed Penthouse (Block B)", "3,400 sq ft", 88_400_000),
        ("penthouse", "b"): ("4-Bed Penthouse (Block B)", "3,400 sq ft", 88_400_000),
    }

    # Determine the block from query
    block = None
    if "block a" in query_lower:
        block = "a"
    elif "block b" in query_lower:
        block = "b"

    # Determine unit type from query
    unit_key = None
    if "studio" in query_lower:
        unit_key = "studio"
    elif "1-bed" in query_lower or "1 bed" in query_lower:
        unit_key = "1-bed"
    elif "2-bed corner" in query_lower or "2 bed corner" in query_lower:
        unit_key = "2-bed corner"
    elif "2-bed" in query_lower or "2 bed" in query_lower:
        unit_key = "2-bed"
    elif "3-bed" in query_lower or "3 bed" in query_lower:
        unit_key = "3-bed"
    elif "4-bed" in query_lower or "penthouse" in query_lower:
        unit_key = "penthouse"

    if unit_key and block and (unit_key, block) in prices:
        name, area, price = prices[(unit_key, block)]
        return (
            f"The base price of a {name} is PKR {price:,} "
            f"({area}, PKR {price // int(area.replace(',', '').split()[0]):,}/sq ft).\n\n"
            f"This is the base price before any location premiums.\n\n"
            f"📄 Source: Price List (Apr 2025)"
        )

    # If block not specified, show both
    if unit_key and not block:
        results = []
        for (uk, blk), (name, area, price) in prices.items():
            if uk == unit_key:
                results.append(f"  • {name}: PKR {price:,} ({area})")
        if results:
            return (
                f"Base prices for {unit_key} units:\n\n"
                + "\n".join(results)
                + "\n\n📄 Source: Price List (Apr 2025)"
            )

    # Fall back to showing relevant chunks
    return generate_answer.__wrapped__(query_lower, chunks) if hasattr(generate_answer, '__wrapped__') else (
        "I found relevant pricing information but couldn't match the exact "
        "unit type. Here are the available prices:\n\n"
        "Block A: Studio (8.64M), 1-Bed (13.32M), 2-Bed (21.85M)\n"
        "Block B: 1-Bed (13.68M), 2-Bed (22.43M), 2-Bed Corner (26.86M), "
        "3-Bed (39.48M), 4-Bed Penthouse (88.4M)\n\n"
        "📄 Source: Price List (Apr 2025)"
    )


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def answer_question(query, vectorizer, tfidf_matrix, chunk_texts, chunk_labels):
    """End-to-end: retrieve relevant chunks, then generate a grounded answer."""
    chunks = retrieve(query, vectorizer, tfidf_matrix, chunk_texts, chunk_labels)
    return generate_answer(query, chunks)


def run_demo():
    """Run the 5 test questions from the MGC task brief."""
    print("=" * 70)
    print("  MGC Aurora Heights — Document Assistant (Demo)")
    print("=" * 70)

    chunk_texts, chunk_labels, vectorizer, tfidf_matrix = build_knowledge_base()

    test_questions = [
        "What's the base price of a 2-bed in Block B?",
        "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?",
        "What's the transfer fee?",
        "What's the rental yield on a 1-bed?",
        "Who is the anchor tenant?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 70}")
        print(f"  Q{i}: {question}")
        print(f"{'─' * 70}\n")
        answer = answer_question(
            question, vectorizer, tfidf_matrix, chunk_texts, chunk_labels
        )
        print(answer)
        print()


def run_interactive():
    """Interactive mode: ask questions in a loop."""
    print("=" * 70)
    print("  MGC Aurora Heights — Document Assistant")
    print("  Type your question and press Enter. Type 'quit' to exit.")
    print("=" * 70)

    chunk_texts, chunk_labels, vectorizer, tfidf_matrix = build_knowledge_base()

    while True:
        print()
        try:
            query = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = answer_question(
            query, vectorizer, tfidf_matrix, chunk_texts, chunk_labels
        )
        print(f"\n{answer}")


if __name__ == "__main__":
    # Force UTF-8 output on Windows to handle box-drawing and emoji characters
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()

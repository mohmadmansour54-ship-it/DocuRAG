from app.prompts import NOT_FOUND_ANSWER


def test_ingest_stores_chunks_with_metadata(rag, sample_pdf):
    result = rag.ingest_pdf(sample_pdf, "manual.pdf")
    assert result.pages == 2
    assert result.chunks >= 2
    assert not result.already_indexed
    assert rag.store.list_documents()[result.document_id] == {"filename": "manual.pdf", "chunks": result.chunks}


def test_reuploading_same_file_does_not_re_embed(rag, embedder, sample_pdf):
    first = rag.ingest_pdf(sample_pdf, "manual.pdf")
    calls_after_first = len(embedder.calls)
    second = rag.ingest_pdf(sample_pdf, "manual-copy.pdf")
    assert second.already_indexed
    assert second.document_id == first.document_id
    assert len(embedder.calls) == calls_after_first


def test_answer_includes_cited_sources(rag, llm, sample_pdf):
    rag.ingest_pdf(sample_pdf, "manual.pdf")
    response = rag.ask("How long does the warranty last?")

    assert response.found
    assert response.answer == llm.answer
    assert [s.citation for s in response.sources] == [1]
    assert response.sources[0].filename == "manual.pdf"
    assert response.sources[0].page == 2  # the warranty is on page 2
    assert "24 months" in response.sources[0].text


def test_most_relevant_passage_is_first_and_irrelevant_ones_are_dropped(rag, llm, sample_pdf):
    rag.ingest_pdf(sample_pdf, "manual.pdf")
    rag.ask("How long does the warranty last?")
    _, user_prompt = llm.calls[0]
    first_passage = user_prompt.split("[1] ", 1)[1].split("\n")[0]
    assert "24 months" in first_passage
    # The battery passage shares no words with the question, so it falls
    # outside max_distance and is never shown to the model.
    assert "battery" not in user_prompt


def test_unrelated_question_is_refused_without_calling_llm(rag, llm, sample_pdf):
    rag.ingest_pdf(sample_pdf, "manual.pdf")
    response = rag.ask("Who won the football world cup?")
    assert not response.found
    assert response.answer == NOT_FOUND_ANSWER
    assert response.sources == []
    assert llm.calls == []


def test_model_saying_not_found_is_reported_as_not_found(rag, llm, sample_pdf):
    rag.ingest_pdf(sample_pdf, "manual.pdf")
    llm.answer = NOT_FOUND_ANSWER
    response = rag.ask("Does the warranty cover the battery?")
    assert not response.found
    assert response.sources == []


def test_answer_without_citations_falls_back_to_all_passages(rag, llm, sample_pdf):
    rag.ingest_pdf(sample_pdf, "manual.pdf")
    llm.answer = "It lasts 24 months."
    response = rag.ask("How long does the warranty last?")
    assert response.found
    assert len(response.sources) >= 1
    assert [s.citation for s in response.sources] == list(range(1, len(response.sources) + 1))


def test_out_of_range_citations_are_ignored(rag, llm, sample_pdf):
    rag.ingest_pdf(sample_pdf, "manual.pdf")
    llm.answer = "It lasts 24 months [1][9]."
    response = rag.ask("How long does the warranty last?")
    assert [s.citation for s in response.sources] == [1]


def test_empty_store_returns_not_found(rag, llm):
    response = rag.ask("How long does the warranty last?")
    assert not response.found
    assert llm.calls == []


def test_prompt_contains_grounding_rules(rag, llm, sample_pdf):
    rag.ingest_pdf(sample_pdf, "manual.pdf")
    rag.ask("How long does the warranty last?")
    system_prompt, _ = llm.calls[0]
    assert "ONLY" in system_prompt
    assert NOT_FOUND_ANSWER in system_prompt

import hashlib

import chromadb
import ollama
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATABASE_PATH = "./chroma_db"

COLLECTION_NAME = "document_qa_collection"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

ANSWER_MODEL_NAME = "mistral"


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Persistent Local Document Q&A",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    """
    Load and cache the local Sentence Transformer model.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# --------------------------------------------------
# CONNECT TO CHROMADB
# --------------------------------------------------

@st.cache_resource
def connect_to_chromadb():
    """
    Create a persistent ChromaDB client.

    The database is stored inside the chroma_db folder.
    Stored documents remain available after the app closes.
    """

    client = chromadb.PersistentClient(
        path=DATABASE_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Persistent PDF document chunks"
            ),
            "embedding_model": (
                EMBEDDING_MODEL_NAME
            ),
            "hnsw:space": "cosine"
        }
    )

    return client, collection


# --------------------------------------------------
# PDF PAGE EXTRACTION
# --------------------------------------------------

def extract_pdf_pages(pdf_file):
    """
    Extract machine-readable text from each PDF page.

    Returns:
        page_records:
            List containing document name, page number
            and extracted text.

        total_pages:
            Total number of pages in the PDF.
    """

    reader = PdfReader(pdf_file)

    page_records = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        page_text = page.extract_text()

        if page_text and page_text.strip():
            page_records.append(
                {
                    "document": pdf_file.name,
                    "page": page_number,
                    "text": page_text.strip()
                }
            )

    return page_records, len(reader.pages)


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_text_chunks(
    page_records,
    chunk_size=1000,
    chunk_overlap=200
):
    """
    Split page text into overlapping chunks.

    Each chunk preserves:
    - Document filename
    - PDF page number
    - Page chunk number
    - Character positions
    - Chunk text
    """

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    chunks = []

    step_size = chunk_size - chunk_overlap

    for page_record in page_records:
        page_text = page_record["text"]

        start_position = 0
        page_chunk_number = 1

        while start_position < len(page_text):
            end_position = min(
                start_position + chunk_size,
                len(page_text)
            )

            chunk_text = page_text[
                start_position:end_position
            ].strip()

            if chunk_text:
                chunks.append(
                    {
                        "document": page_record["document"],
                        "page": page_record["page"],
                        "page_chunk": page_chunk_number,
                        "start_character": start_position,
                        "end_character": end_position,
                        "text": chunk_text
                    }
                )

                page_chunk_number += 1

            if end_position >= len(page_text):
                break

            start_position += step_size

    return chunks


# --------------------------------------------------
# UNIQUE CHUNK ID
# --------------------------------------------------

def create_chunk_id(chunk):
    """
    Create a stable ID for a document chunk.

    Processing the same PDF again updates the existing
    records instead of creating duplicates.
    """

    identifier_text = (
        f"{chunk['document']}|"
        f"{chunk['page']}|"
        f"{chunk['page_chunk']}|"
        f"{chunk['text']}"
    )

    identifier_hash = hashlib.sha256(
        identifier_text.encode("utf-8")
    ).hexdigest()

    return identifier_hash


# --------------------------------------------------
# STORE CHUNKS IN CHROMADB
# --------------------------------------------------

def store_chunks_in_chromadb(
    collection,
    embedding_model,
    chunks
):
    """
    Generate embeddings and persist the chunks,
    metadata and vectors in ChromaDB.
    """

    if not chunks:
        raise ValueError(
            "No chunks are available for storage."
        )

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    chunk_ids = [
        create_chunk_id(chunk)
        for chunk in chunks
    ]

    metadatas = []

    for chunk in chunks:
        metadatas.append(
            {
                "document": chunk["document"],
                "page": int(chunk["page"]),
                "page_chunk": int(
                    chunk["page_chunk"]
                ),
                "start_character": int(
                    chunk["start_character"]
                ),
                "end_character": int(
                    chunk["end_character"]
                )
            }
        )

    collection.upsert(
        ids=chunk_ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return len(chunks)


# --------------------------------------------------
# SEARCH CHROMADB
# --------------------------------------------------

def search_chromadb(
    collection,
    embedding_model,
    question,
    top_k=5
):
    """
    Search the persistent ChromaDB collection using
    the semantic embedding of the user's question.
    """

    if not question.strip():
        raise ValueError(
            "The question cannot be empty."
        )

    available_records = collection.count()

    if available_records == 0:
        return []

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    result_count = min(
        int(top_k),
        available_records
    )

    query_results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    returned_documents = (
        query_results["documents"][0]
    )

    returned_metadatas = (
        query_results["metadatas"][0]
    )

    returned_distances = (
        query_results["distances"][0]
    )

    search_results = []

    for position, document_text in enumerate(
        returned_documents,
        start=1
    ):
        metadata = returned_metadatas[
            position - 1
        ]

        distance = float(
            returned_distances[position - 1]
        )

        similarity_score = max(
            0.0,
            1.0 - distance
        )

        search_results.append(
            {
                "rank": position,
                "document": metadata["document"],
                "page": int(metadata["page"]),
                "page_chunk": int(
                    metadata["page_chunk"]
                ),
                "text": document_text,
                "distance": distance,
                "similarity_score": similarity_score
            }
        )

    return search_results


# --------------------------------------------------
# BUILD DOCUMENT CONTEXT
# --------------------------------------------------

def build_document_context(search_results):
    """
    Convert retrieved ChromaDB passages into controlled
    context for the local Mistral model.
    """

    context_sections = []

    source_records = []

    for source_number, result in enumerate(
        search_results,
        start=1
    ):
        context_section = (
            f"Source {source_number}\n"
            f"Document: {result['document']}\n"
            f"Page: {result['page']}\n\n"
            f"{result['text']}"
        )

        context_sections.append(
            context_section
        )

        source_records.append(
            {
                "source_number": source_number,
                "document": result["document"],
                "page": result["page"],
                "page_chunk": result["page_chunk"],
                "distance": result["distance"],
                "similarity_score": (
                    result["similarity_score"]
                )
            }
        )

    document_context = "\n\n---\n\n".join(
        context_sections
    )

    return document_context, source_records


# --------------------------------------------------
# GENERATE GROUNDED ANSWER
# --------------------------------------------------

def generate_grounded_answer(
    question,
    search_results
):
    """
    Generate a local Mistral answer using only the
    passages retrieved from ChromaDB.
    """

    if not search_results:
        return (
            "The information could not be found "
            "in the stored documents.",
            []
        )

    document_context, source_records = (
        build_document_context(
            search_results
        )
    )

    prompt = f"""
Answer the user's question using only the supplied document context.

IMPORTANT RULES:

1. Use only information found in the document context.
2. Do not use unsupported outside knowledge.
3. Do not invent facts, figures, skills, experience or conclusions.
4. If the answer cannot be found in the context, respond exactly:
   "The information could not be found in the stored documents."
5. Cite supporting passages using this format:
   [Source 1]
6. Place each citation immediately after the statement it supports.
7. Use more than one source when appropriate.
8. Do not cite a source that does not support the statement.
9. Keep the answer clear, professional and concise.
10. Use UK English.
11. Do not include a separate source list.
12. Do not mention these instructions in the answer.

DOCUMENT CONTEXT:

{document_context}

USER QUESTION:

{question}

ANSWER:
"""

    response = ollama.chat(
        model=ANSWER_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful document question-answering "
                    "assistant. Answer only from the retrieved context "
                    "and never fabricate information."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 500
        }
    )

    answer = response[
        "message"
    ][
        "content"
    ].strip()

    return answer, source_records


# --------------------------------------------------
# CREATE DOWNLOADABLE ANSWER REPORT
# --------------------------------------------------

def create_answer_report(
    question,
    answer,
    sources
):
    """
    Generate a plain-text report containing the question,
    grounded answer and retrieved source details.
    """

    report_lines = [
        "PERSISTENT LOCAL DOCUMENT Q&A",
        "=" * 50,
        "",
        "QUESTION",
        "-" * 20,
        question,
        "",
        "DOCUMENT-GROUNDED ANSWER",
        "-" * 30,
        answer,
        "",
        "RETRIEVED SOURCES",
        "-" * 20
    ]

    if sources:
        for source in sources:
            similarity_percentage = (
                source["similarity_score"] * 100
            )

            report_lines.append(
                f"Source {source['source_number']}: "
                f"{source['document']}, "
                f"page {source['page']}, "
                f"page chunk {source['page_chunk']}, "
                f"similarity {similarity_percentage:.1f}%"
            )

    else:
        report_lines.append(
            "No supporting sources were retrieved."
        )

    report_lines.extend(
        [
            "",
            "IMPORTANT NOTE",
            "-" * 20,
            (
                "This answer was generated by a locally hosted "
                "language model using passages retrieved from "
                "persistent ChromaDB storage. Review the answer "
                "against the source passages before using it."
            )
        ]
    )

    return "\n".join(
        report_lines
    )


# --------------------------------------------------
# CLEAR PERSISTENT DATABASE
# --------------------------------------------------

def clear_collection(client):
    """
    Delete and recreate the persistent collection.
    """

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:
        pass

    new_collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Persistent PDF document chunks"
            ),
            "embedding_model": (
                EMBEDDING_MODEL_NAME
            ),
            "hnsw:space": "cosine"
        }
    )

    return new_collection


# --------------------------------------------------
# RESET QUESTION RESULTS
# --------------------------------------------------

def reset_question_results():
    """
    Clear the previous search and answer results.
    """

    keys_to_remove = [
        "chroma_search_results",
        "chroma_question",
        "generated_answer",
        "answer_sources"
    ]

    for key in keys_to_remove:
        st.session_state.pop(
            key,
            None
        )


# --------------------------------------------------
# APPLICATION HEADING
# --------------------------------------------------

st.title(
    "Persistent Local Document Q&A Assistant"
)

st.write(
    "Upload PDF documents, store their embeddings "
    "persistently in ChromaDB and ask questions using "
    "a locally hosted Mistral language model."
)

st.info(
    "Stored documents remain searchable after the "
    "application is closed and restarted. Answers are "
    "generated locally using retrieved document passages."
)


# --------------------------------------------------
# LOAD LOCAL SERVICES
# --------------------------------------------------

try:
    with st.spinner(
        "Loading the local embedding model..."
    ):
        embedding_model = (
            load_embedding_model()
        )

    chroma_client, collection = (
        connect_to_chromadb()
    )

except Exception as error:
    st.error(
        "The local AI services could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.header(
        "RAG Settings"
    )

    chunk_size = st.number_input(
        "Chunk size",
        min_value=300,
        max_value=3000,
        value=1000,
        step=100,
        help=(
            "Maximum characters stored in each chunk."
        )
    )

    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=1000,
        value=200,
        step=50,
        help=(
            "Characters repeated between neighbouring chunks."
        )
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "Number of relevant passages retrieved "
            "and supplied to Mistral."
        )
    )

    if chunk_overlap < chunk_size:
        st.write(
            f"Chunk step size: "
            f"{chunk_size - chunk_overlap}"
        )

    else:
        st.error(
            "Chunk overlap must be smaller "
            "than chunk size."
        )

    st.divider()

    st.metric(
        "Stored Chunks",
        collection.count()
    )

    st.caption(
        "Persistent database location"
    )

    st.code(
        DATABASE_PATH
    )

    st.caption(
        "Embedding model"
    )

    st.code(
        "all-MiniLM-L6-v2"
    )

    st.caption(
        "Answer model"
    )

    st.code(
        ANSWER_MODEL_NAME
    )


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:
    st.success(
        f"{len(uploaded_files)} document(s) selected."
    )

    for uploaded_file in uploaded_files:
        st.write(
            f"- {uploaded_file.name}"
        )


# --------------------------------------------------
# PROCESS AND STORE DOCUMENTS
# --------------------------------------------------

if st.button(
    "Process and Store Documents",
    type="primary",
    use_container_width=True
):
    if not uploaded_files:
        st.error(
            "Please upload at least one PDF document."
        )

    elif chunk_overlap >= chunk_size:
        st.error(
            "Chunk overlap must be smaller than chunk size."
        )

    else:
        try:
            all_chunks = []

            total_pdf_pages = 0
            pages_with_text = 0

            with st.spinner(
                "Extracting and chunking documents..."
            ):
                for uploaded_file in uploaded_files:
                    (
                        page_records,
                        pdf_page_count
                    ) = extract_pdf_pages(
                        uploaded_file
                    )

                    total_pdf_pages += (
                        pdf_page_count
                    )

                    pages_with_text += len(
                        page_records
                    )

                    document_chunks = (
                        create_text_chunks(
                            page_records,
                            chunk_size=int(
                                chunk_size
                            ),
                            chunk_overlap=int(
                                chunk_overlap
                            )
                        )
                    )

                    all_chunks.extend(
                        document_chunks
                    )

            if not all_chunks:
                st.error(
                    "No machine-readable text could be "
                    "extracted from the uploaded documents."
                )

                st.info(
                    "The PDFs may contain scanned images "
                    "rather than selectable text."
                )

            else:
                with st.spinner(
                    "Generating embeddings and storing "
                    "them in ChromaDB..."
                ):
                    stored_count = (
                        store_chunks_in_chromadb(
                            collection,
                            embedding_model,
                            all_chunks
                        )
                    )

                reset_question_results()

                st.success(
                    "The documents were stored successfully."
                )

                (
                    summary_column_1,
                    summary_column_2,
                    summary_column_3
                ) = st.columns(3)

                with summary_column_1:
                    st.metric(
                        "PDF Pages",
                        total_pdf_pages
                    )

                with summary_column_2:
                    st.metric(
                        "Pages With Text",
                        pages_with_text
                    )

                with summary_column_3:
                    st.metric(
                        "Chunks Processed",
                        stored_count
                    )

                st.write(
                    f"Total chunks currently stored: "
                    f"{collection.count()}"
                )

                st.info(
                    "Refresh the browser to update the "
                    "Stored Chunks metric in the sidebar."
                )

        except Exception as error:
            st.error(
                "Document storage failed."
            )

            st.code(
                str(error)
            )


# --------------------------------------------------
# QUESTION INTERFACE
# --------------------------------------------------

st.divider()

st.subheader(
    "Ask Questions About Stored Documents"
)

question = st.text_input(
    "Enter a question",
    placeholder=(
        "Example: What experience does the candidate "
        "have with Power BI?"
    )
)


if st.button(
    "Ask Stored Documents",
    type="primary",
    use_container_width=True
):
    if collection.count() == 0:
        st.error(
            "The persistent collection is empty. "
            "Upload and store documents first."
        )

    elif not question.strip():
        st.error(
            "Please enter a question."
        )

    else:
        try:
            with st.spinner(
                "Searching the persistent vector database..."
            ):
                search_results = search_chromadb(
                    collection=collection,
                    embedding_model=embedding_model,
                    question=question,
                    top_k=int(
                        top_k
                    )
                )

            st.session_state[
                "chroma_search_results"
            ] = search_results

            st.session_state[
                "chroma_question"
            ] = question

            with st.spinner(
                "Mistral is generating a grounded answer. "
                "This may take a few minutes..."
            ):
                (
                    generated_answer,
                    answer_sources
                ) = generate_grounded_answer(
                    question,
                    search_results
                )

            st.session_state[
                "generated_answer"
            ] = generated_answer

            st.session_state[
                "answer_sources"
            ] = answer_sources

            st.success(
                "The document-grounded answer was "
                "generated successfully."
            )

        except Exception as error:
            st.error(
                "The question-answering process failed."
            )

            st.code(
                str(error)
            )

            st.info(
                "Confirm that Ollama is running and that "
                "Mistral appears in 'ollama list'."
            )


# --------------------------------------------------
# DISPLAY GROUNDED ANSWER
# --------------------------------------------------

if st.session_state.get(
    "generated_answer"
):
    generated_answer = st.session_state[
        "generated_answer"
    ]

    answer_sources = st.session_state[
        "answer_sources"
    ]

    stored_question = st.session_state[
        "chroma_question"
    ]

    st.divider()

    st.subheader(
        "Document-Grounded Answer"
    )

    st.markdown(
        f"**Question:** {stored_question}"
    )

    st.markdown(
        generated_answer
    )

    st.warning(
        "Review the answer against the supporting passages. "
        "A local language model can still misinterpret "
        "retrieved information."
    )

    answer_report = create_answer_report(
        stored_question,
        generated_answer,
        answer_sources
    )

    st.download_button(
        label="Download Answer and Sources",
        data=answer_report,
        file_name="persistent_document_answer.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.subheader(
        "Sources Supplied to Mistral"
    )

    for source in answer_sources:
        similarity_percentage = (
            source["similarity_score"] * 100
        )

        st.write(
            f"**Source {source['source_number']}:** "
            f"{source['document']}, "
            f"page {source['page']}, "
            f"page chunk {source['page_chunk']} "
            f"(similarity: "
            f"{similarity_percentage:.1f}%)"
        )


# --------------------------------------------------
# DISPLAY RETRIEVED PASSAGES
# --------------------------------------------------

if st.session_state.get(
    "chroma_search_results"
):
    search_results = st.session_state[
        "chroma_search_results"
    ]

    stored_question = st.session_state[
        "chroma_question"
    ]

    st.divider()

    st.subheader(
        "Supporting Document Passages"
    )

    st.markdown(
        f"**Retrieved for:** {stored_question}"
    )

    st.info(
        "Inspect these retrieved passages to confirm "
        "that the generated answer is properly supported."
    )

    for result in search_results:
        similarity_percentage = (
            result["similarity_score"] * 100
        )

        heading = (
            f"Source {result['rank']} | "
            f"{result['document']} | "
            f"Page {result['page']} | "
            f"Similarity {similarity_percentage:.1f}%"
        )

        with st.expander(
            heading,
            expanded=(
                result["rank"] == 1
            )
        ):
            (
                result_column_1,
                result_column_2
            ) = st.columns(2)

            with result_column_1:
                st.write(
                    f"Document: "
                    f"{result['document']}"
                )

                st.write(
                    f"PDF page: "
                    f"{result['page']}"
                )

                st.write(
                    f"Page chunk: "
                    f"{result['page_chunk']}"
                )

            with result_column_2:
                st.write(
                    f"Chroma distance: "
                    f"{result['distance']:.4f}"
                )

                st.write(
                    f"Similarity score: "
                    f"{result['similarity_score']:.4f}"
                )

                st.write(
                    f"Chunk length: "
                    f"{len(result['text'])}"
                )

            st.markdown(
                "### Retrieved Text"
            )

            st.write(
                result["text"]
            )


# --------------------------------------------------
# DATABASE MANAGEMENT
# --------------------------------------------------

st.divider()

st.subheader(
    "Database Management"
)

st.warning(
    "Clearing the database permanently removes all "
    "stored document chunks and embeddings."
)


if st.button(
    "Clear Persistent Database",
    use_container_width=True
):
    try:
        collection = clear_collection(
            chroma_client
        )

        reset_question_results()

        st.cache_resource.clear()

        st.success(
            "The persistent database was cleared. "
            "Refresh the page to continue."
        )

    except Exception as error:
        st.error(
            "The database could not be cleared."
        )

        st.code(
            str(error)
        )


# --------------------------------------------------
# COMPLETE PIPELINE
# --------------------------------------------------

st.divider()

st.subheader(
    "Complete Persistent Local RAG Pipeline"
)

st.code(
    """
PDF Documents
      ↓
Page-Aware Text Extraction
      ↓
Overlapping Text Chunks
      ↓
384-Dimensional Embeddings
      ↓
Persistent ChromaDB Storage
      ↓
Application Closes and Restarts
      ↓
Question Embedding
      ↓
Persistent Semantic Retrieval
      ↓
Top Relevant Document Passages
      ↓
Controlled Document Context
      ↓
Local Mistral Language Model
      ↓
Grounded Answer With Sources
    """
)

st.success(
    "This application now combines persistent vector "
    "storage, semantic retrieval and local Mistral "
    "answer generation."
)

st.info(
    "This is a proof-of-concept application. All generated "
    "answers must be checked against the retrieved sources."
)
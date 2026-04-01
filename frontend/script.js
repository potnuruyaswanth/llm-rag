const API_BASE_URL = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const askBtn = document.getElementById("askBtn");
const uploadStatus = document.getElementById("uploadStatus");
const selectedFileElement = document.getElementById("selectedFile");
const questionInput = document.getElementById("question");
const answerElement = document.getElementById("answer");
const sourcesElement = document.getElementById("sources");

uploadBtn.addEventListener("click", uploadPDF);
askBtn.addEventListener("click", askQuestion);
fileInput.addEventListener("change", updateSelectedFile);

let documentReady = false;
let activeDocumentName = "";

function updateSelectedFile() {
    const file = fileInput.files[0];
    selectedFileElement.textContent = file ? `Selected file: ${file.name}` : "No file selected.";
}

async function uploadPDF() {
    const file = fileInput.files[0];

    if (!file) {
        uploadStatus.textContent = "Choose a PDF file first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    documentReady = false;
    activeDocumentName = "";
    askBtn.disabled = true;
    uploadBtn.disabled = true;
    uploadStatus.textContent = "Uploading and indexing your PDF. The first upload can take a little longer while models load.";
    answerElement.textContent = "Your answer will appear here.";
    sourcesElement.innerHTML = '<p class="muted">No sources retrieved yet.</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: "POST",
            body: formData
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        const documentInfo = data.document;
        documentReady = true;
        activeDocumentName = documentInfo.filename;
        askBtn.disabled = false;
        uploadStatus.textContent = `${documentInfo.filename} indexed with ${documentInfo.chunks_created} chunks across ${documentInfo.pages_indexed} pages.`;
    } catch (error) {
        uploadStatus.textContent = formatErrorMessage(error, "Upload failed.");
    } finally {
        uploadBtn.disabled = false;
    }
}

async function askQuestion() {
    const query = questionInput.value.trim();
    if (!query) {
        answerElement.textContent = "Enter a question first.";
        return;
    }
    if (!documentReady) {
        answerElement.textContent = "Upload and index a PDF before asking a question.";
        return;
    }

    askBtn.disabled = true;
    answerElement.textContent = "Generating answer...";
    sourcesElement.innerHTML = `<p class="muted">Searching ${activeDocumentName || "the document"}...</p>`;

    try {
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query,
                top_k: 3
            })
        });

        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.detail || "Failed to generate an answer.");
        }

        answerElement.textContent = data.answer;
        renderSources(data.sources || []);
    } catch (error) {
        answerElement.textContent = formatErrorMessage(error, "Failed to generate an answer.");
        sourcesElement.innerHTML = '<p class="muted">No sources available.</p>';
    } finally {
        askBtn.disabled = false;
    }
}

function renderSources(sources) {
    if (!sources.length) {
        sourcesElement.innerHTML = '<p class="muted">No sources available.</p>';
        return;
    }

    sourcesElement.innerHTML = sources
        .map(
            (source) => `
                <article class="source-item">
                    <p class="source-title">${source.source} - page ${source.page}</p>
                    <p>${source.preview}...</p>
                </article>
            `
        )
        .join("");
}

function formatErrorMessage(error, fallbackMessage) {
    if (error instanceof TypeError) {
        return "Cannot reach the backend. Make sure uvicorn is running on http://127.0.0.1:8000.";
    }
    return error.message || fallbackMessage;
}

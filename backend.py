import os
import uuid
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Make sure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.6-27b")

client = Groq(api_key=GROQ_API_KEY)

# RAG Setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=".chromadb")
collection = chroma_client.get_or_create_collection(name="syllabus")
REFERENCES_DIR = "references"
os.makedirs(REFERENCES_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/subjects")
async def get_subjects():
    if not os.path.exists(REFERENCES_DIR):
        return {"subjects": []}
    subjects = [d for d in os.listdir(REFERENCES_DIR) if os.path.isdir(os.path.join(REFERENCES_DIR, d))]
    return {"subjects": subjects}

@app.post("/refresh-index")
async def refresh_index():
    if not os.path.exists(REFERENCES_DIR):
        return {"status": "No references directory found"}
    
    global collection
    try:
        chroma_client.delete_collection("syllabus")
    except:
        pass
    collection = chroma_client.create_collection("syllabus")
    
    docs = []
    metadatas = []
    ids = []
    
    for subject in os.listdir(REFERENCES_DIR):
        subj_path = os.path.join(REFERENCES_DIR, subject)
        if not os.path.isdir(subj_path):
            continue
            
        for file in os.listdir(subj_path):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(subj_path, file)
                try:
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted + "\n"
                        
                        chunk_size = 1000
                        overlap = 200
                        start = 0
                        while start < len(text):
                            end = start + chunk_size
                            chunk = text[start:end]
                            if chunk.strip():
                                docs.append(chunk)
                                metadatas.append({"subject": subject, "file": file})
                                ids.append(str(uuid.uuid4()))
                            start = end - overlap
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    if docs:
        embeddings = embedder.encode(docs).tolist()
        collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
    return {"status": "success", "chunks_indexed": len(docs)}

@app.post("/solve")
async def solve_question(request: Request):
    data = await request.json()
    base64_image = data.get("image")
    active_subjects = data.get("active_subjects", [])
    
    def generate():
        active_model = MODEL_NAME
        prompt = "Please read the question in this image and provide a clear, concise answer or explanation, if it is a MCQ then just tell the option letter (like 'A' is the answer) and also if the image is unclear you can tell the user that the question is unclear and to send the image again ('the image is unclear, please send it again')"
        extracted_text = ""
        
        # --- RAG Phase (Cloud-friendly Groq API Extraction) ---
        if active_subjects and collection.count() > 0:
            yield f"🔍 **Step 1:** Reading the question from the image...\n\n"
            
            try:
                extract_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all the text and identify the core question in this image. Only output the extracted text and question, nothing else."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
                
                # We use the Groq API again, but we STREAM it so the user isn't stuck waiting
                extraction_stream = client.chat.completions.create(
                    model=active_model,
                    messages=extract_messages,
                    temperature=0.2,
                    max_completion_tokens=500,
                    stream=True
                )
                
                for chunk in extraction_stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        extracted_text += content
                        yield f"> _{content}_" if len(extracted_text) == len(content) else content
                
                yield f"\n\n📚 **Step 2:** Searching syllabus for matching concepts...\n\n"
                
                if extracted_text.strip():
                    # Search ChromaDB
                    query_embedding = embedder.encode(extracted_text).tolist()
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3,
                        where={"subject": {"$in": active_subjects}}
                    )
                    
                    context = "\n\n".join(results['documents'][0]) if results['documents'] and results['documents'][0] else "No relevant context found."
                    
                    prompt = (
                        f"You are a helpful solver. Use the following context from the user's syllabus ({', '.join(active_subjects)}) to help answer the question in the image.\n\n"
                        f"Context from syllabus:\n{context}\n\n"
                        "If the context from the syllabus is not enough for answering the specific question, you can use your general knowledge, "
                        "but try to stick to the syllabus's context and vocabulary if possible.\n"
                        "Please read the question in the image and provide a clear, concise answer or explanation."
                    )
                    
                    yield f"💡 **Step 3:** Generating final answer using syllabus context...\n\n---\n\n"
                else:
                    yield f"⚠️ No text found in image. Falling back to standard mode...\n\n"
                    
            except Exception as e:
                print("Cloud extraction failed:", e)
                yield f"⚠️ Cloud extraction failed: {str(e)}\nFalling back to standard mode...\n\n"

        # --- Final Generation Phase ---
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
        
        full_response = ""
        try:
            completion = client.chat.completions.create(
                model=active_model,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=2048,
                top_p=0.95,
                stream=True,
                stop=None
            )
        except Exception as e:
            fallback_model = "qwen/qwen3.8-27b"
            yield f"⚠️ Primary model ({active_model}) failed. Switching to fallback ({fallback_model})...\n\n"
            active_model = fallback_model
            try:
                completion = client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    temperature=0.6,
                    max_completion_tokens=2048,
                    top_p=0.95,
                    stream=True,
                    stop=None
                )
            except Exception as fallback_err:
                yield f"❌ Fallback model also failed: {str(fallback_err)}"
                return

        try:
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content
                    
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("solve_logs.md", "a", encoding="utf-8") as f:
                f.write(f"## {timestamp} (Model: {active_model})\n\n**RAG Extracted Text:**\n{extracted_text if active_subjects else 'None'}\n\n**Final Answer:**\n{full_response.strip()}\n\n---\n\n")
                
        except Exception as stream_err:
            yield f"\n\n❌ Stream interrupted: {str(stream_err)}"

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8080, reload=True)

import os
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/solve")
async def solve_question(request: Request):
    data = await request.json()
    base64_image = data.get("image")
    
    prompt = """This is for my Semester 1 English Literature exam. The questions will be related to these kinds of topics: Shakespeare's Sonnet 18, Robert Frost, Aristotle's Rhetoric, Martin Luther King's 'I Have a Dream', William Blake's 'The Tyger', Ted Hughes, Sarojini Naidu, Nissim Ezekiel, Bertrand Russell, Swami Vivekananda, or Vijay Tendulkar's 'Silence! The Court is in Session'. 
    
    Please read the question in the image and find the answer within this specific region of English Literature. Provide a clear, academic answer. If it is an MCQ, just tell me the correct option letter. If the image is unclear, tell me to send it again."""        
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    def generate():
        full_response = ""
        active_model = MODEL_NAME
        
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
                f.write(f"## {timestamp} (Model: {active_model})\n\n**Final Answer:**\n{full_response.strip()}\n\n---\n\n")
                
        except Exception as stream_err:
            yield f"\n\n❌ Stream interrupted: {str(stream_err)}"

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8080, reload=True)

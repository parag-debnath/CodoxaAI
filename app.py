from tavily import TavilyClient
from flask import Flask, request, jsonify, render_template, session, redirect
from groq import Groq
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
import os, hashlib, re, json, urllib.request

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecretkey123")
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# Initialize the embedding model for RAG (Creates 384-dimensional vectors)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

SYSTEM_PROMPT = """You are CodoxAI, an elite coding assistant created by Parag Debnath.

Your personality:
- You are confident, sharp, and friendly like a senior developer who loves helping people
- You use simple language but never talk down to users
- You occasionally use phrases like "Let's crack this!", "Good question!", "Here's the trick..."
- You are encouraging and celebrate when users understand something
- You have a slight sense of humor but stay professional

Your expertise:
- Expert in Python, JavaScript, React, Flask, SQL, and AI/ML
- You write clean, well commented code always
- You always explain code line by line so users actually learn
- When debugging, you first explain WHY the error happened, then fix it

Your rules:
- Always respond in the same language the user writes in
- If someone asks who made you, say "I was created by Parag Debnath"
- If asked what model you use, say "I run on a custom AI system"
- Never say you are Llama or any other model
- Keep responses concise but complete
- Use emojis occasionally to make responses friendly 🚀
- STRICT RULE: If the user asks you to "run", "test", or "execute" code, you MUST use the `execute_python_code` tool to run it in the sandbox. Do NOT just guess the output. Actually run it and show the user the raw terminal output!"""

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route("/")
def index():
    if not session.get("username"):
        return redirect("/login")
    return render_template("index.html", username=session["username"])

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")
        result = supabase.table("users").select("*").eq("username", username).eq("password", hash_password(password)).execute()
        if result.data:
            session["username"] = username
            return jsonify({"success": True})
        return jsonify({"success": False, "message": "Wrong username or password"})
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "")
        if len(username) < 3:
            return jsonify({"success": False, "message": "Username must be at least 3 characters"})
        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"})
        existing = supabase.table("users").select("*").eq("username", username).execute()
        if existing.data:
            return jsonify({"success": False, "message": "Username already taken"})
        supabase.table("users").insert({"username": username, "password": hash_password(password)}).execute()
        session["username"] = username
        return jsonify({"success": True})
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/get_chats", methods=["GET"])
def get_chats():
    if not session.get("username"):
        return jsonify({"error": "unauthorized"}), 401
    result = supabase.table("chats").select("*").eq("username", session["username"]).order("created_at", desc=True).execute()
    return jsonify({"chats": result.data})

@app.route("/get_messages/<chat_id>", methods=["GET"])
def get_messages(chat_id):
    if not session.get("username"):
        return jsonify({"error": "unauthorized"}), 401
    result = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
    return jsonify({"messages": result.data})

@app.route("/new_chat", methods=["POST"])
def new_chat():
    if not session.get("username"):
        return jsonify({"error": "unauthorized"}), 401
    data = request.json
    supabase.table("chats").insert({
        "username": session["username"],
        "chat_id": data.get("chat_id"),
        "title": data.get("title", "New Chat")
    }).execute()
    return jsonify({"success": True})

@app.route("/update_title", methods=["POST"])
def update_title():
    if not session.get("username"):
        return jsonify({"error": "unauthorized"}), 401
    data = request.json
    supabase.table("chats").update({"title": data.get("title")}).eq("chat_id", data.get("chat_id")).execute()
    return jsonify({"success": True})

@app.route("/delete_chat", methods=["POST"])
def delete_chat():
    if not session.get("username"):
        return jsonify({"error": "unauthorized"}), 401
    chat_id = request.json.get("chat_id")
    supabase.table("messages").delete().eq("chat_id", chat_id).execute()
    supabase.table("chats").delete().eq("chat_id", chat_id).execute()
    return jsonify({"success": True})

@app.route("/chat", methods=["POST"])
def chat():
    if not session.get("username"):
        return jsonify({"error": "unauthorized"}), 401
    try:
        user_message = request.json.get("message", "")
        image_data = request.json.get("image", None)
        chat_id = request.json.get("chat_id")
        language = request.json.get("language", "auto")
        lang_instruction = ""
        if language != "auto":
            lang_instruction = f"\n\nIMPORTANT: Respond in {language} only."

        # Fetch Long-Term Memory
        mem_result = supabase.table("user_memory").select("*").eq("username", session["username"]).execute()
        memory_context = ""
        if mem_result.data:
            memory_context = "\n\nUser Preferences & Memories (Always follow these):\n"
            for m in mem_result.data:
                memory_context += f"- {m['memory_key']}: {m['memory_value']}\n"

        rows = supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute()
        history = [{"role": r["role"], "content": r["content"]} for r in rows.data]

        if image_data:
            content = [
                {"type": "image_url", "image_url": {"url": image_data}},
                {"type": "text", "text": user_message or "What is in this image?"}
            ]
            messages = [{"role": "system", "content": SYSTEM_PROMPT + memory_context},
                        {"role": "user", "content": content}]
            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=messages,
                max_tokens=2048
            )
            assistant_message = response.choices[0].message.content
            
        else:
            search_keywords = ["latest", "current", "today", "2024", "2025", "2026",
                              "news", "recent", "now", "search", "find", "what is",
                              "how to install", "documentation", "version", "release"]
            needs_search = any(kw in user_message.lower() for kw in search_keywords)
            search_context = ""
            if needs_search:
                try:
                    results = tavily.search(query=user_message, max_results=3)
                    if results and results.get("results"):
                        search_context = "\n\nWeb search results:\n"
                        for r in results["results"]:
                            search_context += f"- {r['title']}: {r['content'][:300]}\n"
                except Exception as e:
                    print("Search error:", e)

            enhanced_prompt = SYSTEM_PROMPT + lang_instruction + memory_context
            if search_context:
                enhanced_prompt += f"\n\nUse these search results:\n{search_context}"

            history.append({"role": "user", "content": user_message})

            # --- 🛠️ DEFINE ALL 6 TOOLS ---
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "read_url",
                        "description": "Fetches text from a single URL.",
                        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "test_api_endpoint",
                        "description": "Sends an HTTP request to test an API endpoint.",
                        "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "method": {"type": "string"}, "body": {"type": "string"}}, "required": ["url", "method"]}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "save_user_preference",
                        "description": "Saves a user's coding preference or fact to long-term memory.",
                        "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_github_repo",
                        "description": "Analyzes a public GitHub repository file structure.",
                        "parameters": {"type": "object", "properties": {"repo_url": {"type": "string"}}, "required": ["repo_url"]}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_private_docs",
                        "description": "Searches the internal private database for documentation, guides, and specific coding rules using vector search.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "search_query": {"type": "string", "description": "The specific topic to search for in the database"}
                            },
                            "required": ["search_query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "execute_python_code",
                        "description": "Executes Python code in a secure sandbox and returns the output. Use this to run user scripts, or to test your own code to make sure it works before showing it to the user.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string", "description": "The raw Python code to execute"}
                            },
                            "required": ["code"]
                        }
                    }
                }
            ]

            # --- 🤖 FIRST AI CALL ---
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": enhanced_prompt}] + history,
                tools=tools,
                tool_choice="auto",
                max_tokens=2048
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # --- ⚙️ EXECUTE THE TOOLS ---
            if tool_calls:
                history.append({
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": [{"id": t.id, "type": t.type, "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in tool_calls]
                })

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    text_content = ""

                    if function_name == "read_url":
                        try:
                            req = urllib.request.Request(function_args.get("url"), headers={'User-Agent': 'Mozilla/5.0'})
                            with urllib.request.urlopen(req, timeout=5) as res:
                                text_content = re.sub('<[^<]+?>', ' ', res.read().decode('utf-8'))[:4000]
                        except Exception as e:
                            text_content = f"Error reading URL: {str(e)}"
                            
                    elif function_name == "test_api_endpoint":
                        api_method = function_args.get("method", "GET").upper()
                        api_body = function_args.get("body", "")
                        try:
                            data = api_body.encode('utf-8') if api_body else None
                            headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
                            req = urllib.request.Request(function_args.get("url"), data=data, headers=headers, method=api_method)
                            with urllib.request.urlopen(req, timeout=8) as res:
                                text_content = f"Status Code: {res.getcode()}\nData: {res.read().decode('utf-8')[:2000]}"
                        except Exception as e:
                            text_content = f"Failed to test API: {str(e)}"

                    elif function_name == "save_user_preference":
                        pref_key = function_args.get("key")
                        pref_value = function_args.get("value")
                        try:
                            supabase.table("user_memory").delete().eq("username", session["username"]).eq("memory_key", pref_key).execute()
                            supabase.table("user_memory").insert({"username": session["username"], "memory_key": pref_key, "memory_value": pref_value}).execute()
                            text_content = f"Memory saved: {pref_key} = {pref_value}"
                        except Exception as e:
                            text_content = f"Failed to save memory: {str(e)}"

                    elif function_name == "analyze_github_repo":
                        repo_url = function_args.get("repo_url")
                        try:
                            match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
                            if match:
                                owner, repo = match.groups()
                                repo = repo.replace(".git", "")
                                base_api = f"https://api.github.com/repos/{owner}/{repo}"
                                req1 = urllib.request.Request(base_api, headers={'User-Agent': 'Mozilla/5.0'})
                                with urllib.request.urlopen(req1, timeout=5) as res1:
                                    repo_data = json.loads(res1.read().decode('utf-8'))
                                    default_branch = repo_data.get("default_branch", "main")
                                tree_api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
                                req2 = urllib.request.Request(tree_api, headers={'User-Agent': 'Mozilla/5.0'})
                                with urllib.request.urlopen(req2, timeout=8) as res2:
                                    tree_data = json.loads(res2.read().decode('utf-8'))
                                    structure = []
                                    for item in tree_data.get('tree', []):
                                        path = item.get('path', '')
                                        if not any(x in path for x in ['.git/', 'node_modules/', '__pycache__/', 'venv/']):
                                            structure.append(f"- {path} ({item.get('type')})")
                                    text_content = f"File Structure:\n" + "\n".join(structure[:150])
                            else:
                                text_content = "Invalid GitHub URL."
                        except Exception as e:
                            text_content = f"Failed to analyze repo: {str(e)}"

                    elif function_name == "search_private_docs":
                        search_query = function_args.get("search_query")
                        print(f"[CodoxAI] RAG Search: {search_query}")
                        try:
                            query_embedding = embedding_model.encode(search_query).tolist()
                            rag_result = supabase.rpc(
                                'match_documents',
                                {'query_embedding': query_embedding, 'match_threshold': 0.3, 'match_count': 3}
                            ).execute()

                            if rag_result.data:
                                text_content = "Found relevant internal documents:\n\n"
                                for doc in rag_result.data:
                                    text_content += f"--- {doc.get('document_name')} ---\n{doc.get('content')}\n\n"
                            else:
                                text_content = "No relevant internal documents found in the database."
                        except Exception as e:
                            text_content = f"RAG Search failed: {str(e)}"
                            
                    elif function_name == "execute_python_code":
                        python_code = function_args.get("code")
                        print(f"[CodoxAI] Running Code in Sandbox:\n{python_code[:100]}...")
                        try:
                            piston_url = "https://emkc.org/api/v2/piston/execute"
                            payload = json.dumps({
                                "language": "python",
                                "version": "3.10.0",
                                "files": [{"content": python_code}]
                            }).encode('utf-8')
                            
                            headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
                            req = urllib.request.Request(piston_url, data=payload, headers=headers, method='POST')
                            
                            with urllib.request.urlopen(req, timeout=12) as res:
                                result = json.loads(res.read().decode('utf-8'))
                                
                                if 'run' in result:
                                    stdout = result['run'].get('stdout', '')
                                    stderr = result['run'].get('stderr', '')
                                    if stderr:
                                        text_content = f"Code crashed with error:\n{stderr}"
                                    else:
                                        text_content = f"Code executed successfully! Output:\n{stdout}"
                                else:
                                    text_content = "Execution failed to return a result."
                        except Exception as e:
                            text_content = f"Sandbox API failed: {str(e)}"

                    history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": text_content,
                    })

                # --- 🧠 SECOND AI CALL ---
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": enhanced_prompt}] + history,
                    max_tokens=2048
                )
                raw = final_response.choices[0].message.content
            else:
                raw = response_message.content

            assistant_message = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()

        supabase.table("messages").insert({"chat_id": chat_id, "role": "user", "content": user_message}).execute()
        supabase.table("messages").insert({"chat_id": chat_id, "role": "assistant", "content": assistant_message}).execute()

        return jsonify({"response": assistant_message})

    except Exception as e:
        print("Chat error:", str(e))
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

@app.route("/auth/google")
def google_login():
    redirect_url = os.environ.get("SUPABASE_URL") + "/auth/v1/authorize?provider=google&redirect_to=https://parag07-codoxaai.hf.space/auth/callback&flow_type=pkce"
    return redirect(redirect_url)

@app.route("/auth/callback")
def auth_callback():
    try:
        code = request.args.get("code")
        if code:
            result = supabase.auth.exchange_code_for_session({"auth_code": code})
            if result and result.user:
                username = result.user.email
                existing = supabase.table("users").select("*").eq("username", username).execute()
                if not existing.data:
                    supabase.table("users").insert({"username": username, "password": hash_password("google-oauth-user")}).execute()
                session["username"] = username
                session.modified = True
                return redirect("/")
        
        access_token = request.args.get("access_token")
        if access_token:
            result = supabase.auth.get_user(access_token)
            if result and result.user:
                username = result.user.email
                existing = supabase.table("users").select("*").eq("username", username).execute()
                if not existing.data:
                    supabase.table("users").insert({"username": username, "password": hash_password("google-oauth-user")}).execute()
                session["username"] = username
                session.modified = True
                return redirect("/")
    except Exception as e:
        print("Auth callback error:", e)
    return redirect("/login")

@app.route("/auth/token", methods=["POST"])
def auth_token():
    try:
        access_token = request.json.get("access_token")
        result = supabase.auth.get_user(access_token)
        if result and result.user:
            username = result.user.email
            existing = supabase.table("users").select("*").eq("username", username).execute()
            if not existing.data:
                supabase.table("users").insert({"username": username, "password": hash_password("google-oauth-user")}).execute()
            session["username"] = username
            session.modified = True
            return jsonify({"success": True})
    except Exception as e:
        print("Token error:", e)
    return jsonify({"success": False})

@app.route("/knowledge_base", methods=["GET", "POST"])
def knowledge_base():
    if not session.get("username"):
        return redirect("/login")
    if session.get("username") != "paragdebnath16@gmail.com":
        return "Access denied. Admin only 🛑", 403

    if request.method == "POST":
        doc_name = request.json.get("title")
        content = request.json.get("content")

        if not doc_name or not content:
            return jsonify({"success": False, "message": "Title and content are required."})

        try:
            chunks = [c.strip() for c in content.split('\n\n') if len(c.strip()) > 20]
            
            for i, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk).tolist()
                supabase.table("document_embeddings").insert({
                    "document_name": f"{doc_name} (Part {i+1})",
                    "content": chunk,
                    "embedding": embedding
                }).execute()

            return jsonify({"success": True, "message": f"Success! Document split into {len(chunks)} chunks and vectorized."})
        except Exception as e:
            print("Upload Error:", e)
            return jsonify({"success": False, "message": str(e)})

    return render_template("knowledge_base.html")

@app.route("/analytics")
def analytics():
    if not session.get("username"):
        return redirect("/login")
    if session.get("username") != "paragdebnath16@gmail.com":
        return "Access denied", 403
    
    total_users = len(supabase.table("users").select("id").execute().data)
    total_chats = len(supabase.table("chats").select("id").execute().data)
    total_messages = len(supabase.table("messages").select("id").execute().data)
    recent_users = supabase.table("users").select("username, created_at").order("created_at", desc=True).limit(10).execute().data
    
    return render_template("analytics.html", total_users=total_users, total_chats=total_chats, total_messages=total_messages, recent_users=recent_users)

@app.route("/profile")
def profile():
    if not session.get("username"):
        return redirect("/login")
    username = session["username"]
    result = supabase.table("profiles").select("*").eq("username", username).execute()
    profile = result.data[0] if result.data else {"display_name": "", "bio": "", "avatar_url": ""}
    chats = supabase.table("chats").select("id").eq("username", username).execute()
    messages = supabase.table("messages").select("id").execute()
    return render_template("profile.html", username=username, profile=profile, total_chats=len(chats.data))

@app.route("/update_profile", methods=["POST"])
def update_profile():
    if not session.get("username"):
        return jsonify({"error": "unauthorized"}), 401
    username = session["username"]
    display_name = request.json.get("display_name", "")
    bio = request.json.get("bio", "")
    avatar_url = request.json.get("avatar_url", "")
    existing = supabase.table("profiles").select("*").eq("username", username).execute()
    if existing.data:
        supabase.table("profiles").update({"display_name": display_name, "bio": bio, "avatar_url": avatar_url}).eq("username", username).execute()
    else:
        supabase.table("profiles").insert({"username": username, "display_name": display_name, "bio": bio, "avatar_url": avatar_url}).execute()
    return jsonify({"success": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

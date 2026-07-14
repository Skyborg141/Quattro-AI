from pathlib import Path

route_code = """
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

@app.get("/chat-ui", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(Path("static/index.html").read_text())

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass
"""

main = Path("app/main.py").read_text()

if "/chat-ui" not in main:
    with open("app/main.py", "a") as f:
        f.write(route_code)
    print("Done! Route added.")
else:
    print("Route already exists.")

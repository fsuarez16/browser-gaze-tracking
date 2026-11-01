const http = require("http");
const fs = require("fs");
const path = require("path");
const root = path.join(__dirname, "dist");

http.createServer((request, response) => {
  const pathname = decodeURIComponent(request.url.split("?")[0]);
  const relative = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
  const file = path.resolve(root, relative);
  if (!file.startsWith(path.resolve(root))) return response.writeHead(403).end("forbidden");
  fs.readFile(file, (error, data) => {
    if (error) return response.writeHead(404).end("not found");
    const extension = path.extname(file);
    const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm" };
    response.setHeader("Content-Type", types[extension] || "application/octet-stream");
    response.setHeader("Cache-Control", "no-store");
    response.end(data);
  });
}).listen(4173,"127.0.0.1",()=>console.log("Hybrid Webcam Gaze Tracker: http://127.0.0.1:4173/"));


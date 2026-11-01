import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";

const root = process.cwd();
const types = {".html":"text/html; charset=utf-8",".js":"text/javascript; charset=utf-8",".css":"text/css; charset=utf-8",".wasm":"application/wasm",".task":"application/octet-stream",".json":"application/json; charset=utf-8"};

createServer((request,response) => {
  const pathname = decodeURIComponent(new URL(request.url,"http://localhost").pathname);
  const relative = pathname === "/" ? "index.html" : pathname.replace(/^\/+/,"");
  const file = normalize(join(root,relative));
  if (!file.startsWith(root)) { response.writeHead(403); response.end("Forbidden"); return; }
  try {
    if (!statSync(file).isFile()) throw new Error("Not a file");
    response.writeHead(200,{"Content-Type":types[extname(file)]||"application/octet-stream","Cache-Control":"no-cache"});
    createReadStream(file).pipe(response);
  } catch { response.writeHead(404); response.end("Not found"); }
}).listen(8080,"127.0.0.1",()=>console.log("PCA-Enhanced Webcam Gaze Tracker: http://127.0.0.1:8080/"));

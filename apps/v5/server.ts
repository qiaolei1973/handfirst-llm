/**
 * v5 WS server — runs Trainer, broadcasts epoch events to dashboard.
 */
import { WebSocketServer } from "ws";
import { surfaceData } from "@handfirst/datasets";
import { Trainer } from "./train";

const wss = new WebSocketServer({ port: 3105 });
const { features, labels } = surfaceData(200);
const trainer = new Trainer(features, labels, 16);

console.log("v5 WS server on :3105");

wss.on("connection", (ws) => {
  console.log("dashboard connected");
  trainer.reset();

  let timer: ReturnType<typeof setInterval>;
  timer = setInterval(() => {
    const ev = trainer.step();
    ws.send(JSON.stringify(ev));
    if (ev.stopped || trainer.history.length >= 2000) {
      clearInterval(timer);
      ws.close();
    }
  }, 50);

  ws.on("close", () => clearInterval(timer));
});

// Handle Ctrl+C
process.on("SIGINT", () => { wss.close(); process.exit(); });
process.on("SIGTERM", () => { wss.close(); process.exit(); });

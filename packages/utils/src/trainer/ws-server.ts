import { WebSocketServer, type WebSocket } from 'ws';
import type { ITrainer, InitData, ClientMsg, ServerMsg } from './types';

function tryPort(port: number): Promise<WebSocketServer | null> {
  return new Promise((resolve, reject) => {
    const wss = new WebSocketServer({ port });
    wss.on('listening', () => resolve(wss));
    wss.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE') { wss.close(); resolve(null); }
      else { reject(err); }
    });
  });
}

export class WSServer {
  private _wss!: WebSocketServer;

  constructor(
    port: number,
    private _maxEpochs: number,
    factory: () => ITrainer,
    initData: Omit<InitData, 'params'>,
  ) {
    void this._bind(port, factory, initData);
  }

  private async _bind(port: number, factory: () => ITrainer, initData: Omit<InitData, 'params'>) {
    let actualPort = port;
    for (let i = 0; i < 10; i++) {
      const wss = await tryPort(actualPort);
      if (wss) { this._wss = wss; break; }
      actualPort++;
    }
    if (actualPort !== port) console.log(`  ⚠️  端口 ${port} 已被占用，改用: ${actualPort}`);

    this._wss.on('connection', (ws: WebSocket) => {
      const trainer = factory();
      let timer: ReturnType<typeof setInterval> | null = null;

      const send = (msg: ServerMsg) => { if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(msg)); };
      const stop = () => { if (timer) { clearInterval(timer); timer = null; } };

      const done = () => trainer.history.length >= this._maxEpochs;

      send({ type: 'init', data: { ...initData, params: trainer.params } });

      ws.on('message', (raw) => {
        let msg: ClientMsg;
        try { msg = JSON.parse(raw.toString()); } catch { return; }

        switch (msg.type) {
          case 'play': {
            stop();
            const interval = Math.max(20, Math.floor(100 / msg.speed));
            timer = setInterval(() => {
              const ev = trainer.step() as Record<string, unknown>;
              send({ type: 'epoch', data: { ...ev, epoch: trainer.history.length } });
              if (done()) { stop(); send({ type: 'done' }); }
            }, interval);
            break;
          }
          case 'pause': stop(); break;
          case 'step': {
            const ev = trainer.step() as Record<string, unknown>;
            send({ type: 'epoch', data: { ...ev, epoch: trainer.history.length } });
            if (done()) send({ type: 'done' });
            break;
          }
          case 'reset':
            stop(); trainer.reset();
            send({ type: 'reset', params: trainer.params });
            break;
        }
      });

      ws.on('close', stop);
    });

    console.log(`  🔌 训练 WS 服务启动: ws://localhost:${actualPort}`);
  }

  close(): void { this._wss.close(); }
}

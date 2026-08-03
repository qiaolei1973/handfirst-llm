import { WebSocketServer, type WebSocket } from 'ws';
import type { ITrainer, InitData, ClientMsg, ServerMsg } from './types';

export class WSServer {
  private _wss: WebSocketServer;

  constructor(
    port: number,
    maxEpochs: number,
    factory: () => ITrainer,
    initData: Omit<InitData, 'params'>,
  ) {
    this._wss = new WebSocketServer({ port });

    this._wss.on('connection', (ws: WebSocket) => {
      const trainer = factory();
      let timer: ReturnType<typeof setInterval> | null = null;

      const send = (msg: ServerMsg) => {
        if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(msg));
      };

      const stop = () => {
        if (timer) { clearInterval(timer); timer = null; }
      };

      const done = () => trainer.history.length >= maxEpochs;

      send({ type: 'init', data: { ...initData, params: trainer.params } });

      ws.on('message', (raw) => {
        let msg: ClientMsg;
        try { msg = JSON.parse(raw.toString()); } catch { return; }

        switch (msg.type) {
          case 'play': {
            stop();
            const interval = Math.max(20, Math.floor(100 / msg.speed));
            timer = setInterval(() => {
              if (done()) { stop(); send({ type: 'done' }); return; }
              const ev = trainer.step();
              send({ type: 'epoch', data: { ...ev, epoch: trainer.history.length } });
            }, interval);
            break;
          }
          case 'pause':
            stop();
            break;
          case 'step':
            if (done()) return;
            send({ type: 'epoch', data: { ...trainer.step(), epoch: trainer.history.length } });
            if (done()) send({ type: 'done' });
            break;
          case 'reset':
            stop();
            trainer.reset();
            send({ type: 'reset', params: trainer.params });
            break;
        }
      });

      ws.on('close', stop);
    });

    console.log(`  🔌 训练 WS 服务启动: ws://localhost:${port}`);
  }

  close(): void {
    this._wss.close();
  }
}

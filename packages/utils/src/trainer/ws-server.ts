import { WebSocketServer, type WebSocket } from 'ws';
import type { ITrainer, InitData, ClientMsg, ServerMsg } from './types';

/** 尝试在指定端口创建 WSS，成功返回实例，EADDRINUSE 返回 null */
function tryPort(port: number): Promise<WebSocketServer | null> {
  return new Promise((resolve, reject) => {
    const wss = new WebSocketServer({ port });
    wss.on('listening', () => resolve(wss));
    wss.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'EADDRINUSE') {
        wss.close();
        resolve(null);
      } else {
        reject(err);
      }
    });
  });
}

export class WSServer {
  private _wss!: WebSocketServer;

  constructor(
    port: number,
    maxEpochs: number,
    factory: () => ITrainer,
    initData: Omit<InitData, 'params'>,
  ) {
    void this._bind(port, maxEpochs, factory, initData);
  }

  private async _bind(
    port: number,
    maxEpochs: number,
    factory: () => ITrainer,
    initData: Omit<InitData, 'params'>,
  ) {
    // 尝试绑定端口，被占用则顺延（最多试 10 次）
    let actualPort = port;
    for (let i = 0; i < 10; i++) {
      const wss = await tryPort(actualPort);
      if (wss) { this._wss = wss; break; }
      actualPort++;
    }

    if (actualPort !== port) {
      console.log(`  ⚠️  端口 ${port} 已被占用，改用: ${actualPort}`);
    }

    this._wss.on('connection', (ws: WebSocket) => {
      const trainer = factory();
      let timer: ReturnType<typeof setInterval> | null = null;

      const send = (msg: ServerMsg) => {
        if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(msg));
      };

      const stop = () => {
        if (timer) { clearInterval(timer); timer = null; }
      };

      const done = () =>
	        (trainer.isDone?.() ?? false) || trainer.history.length >= maxEpochs;

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

    console.log(`  🔌 训练 WS 服务启动: ws://localhost:${actualPort}`);
  }

  close(): void {
    this._wss.close();
  }
}

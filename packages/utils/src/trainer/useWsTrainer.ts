'use client';

import { useRef, useEffect, useCallback } from 'react';
import type {
  WsTrainer,
  ClientMsg,
  InitData,
  EpochEvent,
} from './types';

type EpochData = EpochEvent & { epoch: number };

export function useWsTrainer(url: string): WsTrainer {
  const wsRef = useRef<WebSocket | null>(null);

  const stateRef = useRef({
    params: undefined as unknown,
    history: [] as EpochData[],
    dataset: null as InitData | null,
    lastEvent: null as EpochData | null,
  });

  const listenersRef = useRef({
    init: [] as Array<(data: InitData) => void>,
    epoch: [] as Array<(ev: EpochData) => void>,
    done: [] as Array<() => void>,
    reset: [] as Array<() => void>,
  });

  // ---- 连接 ----
  useEffect(() => {
    let ws: WebSocket;

    function connect() {
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        const s = stateRef.current;
        const ls = listenersRef.current;

        switch (msg.type) {
          case 'init': {
            s.dataset = msg.data;
            s.params = msg.data.params;
            s.history = [];
            s.lastEvent = null;
            ls.init.forEach((fn) => fn(msg.data));
            break;
          }
          case 'epoch': {
            const ev = msg.data as EpochData;
            s.params = ev as unknown;
            s.history.push(ev);
            s.lastEvent = ev;
            ls.epoch.forEach((fn) => fn(ev));
            break;
          }
          case 'done': {
            ls.done.forEach((fn) => fn());
            break;
          }
          case 'reset': {
            s.params = msg.params;
            s.history = [];
            s.lastEvent = null;
            ls.reset.forEach((fn) => fn());
            break;
          }
        }
      };

      ws.onclose = () => {
        setTimeout(() => { if (wsRef.current === ws) connect(); }, 3000);
      };
    }

    connect();
    return () => { wsRef.current = null; ws.close(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // ---- 命令 ----
  const send = useCallback((msg: ClientMsg) => {
    wsRef.current?.send(JSON.stringify(msg));
  }, []);

  const step = useCallback(() => send({ type: 'step' }), [send]);
  const play = useCallback((speed: number) => send({ type: 'play', speed }), [send]);
  const pause = useCallback(() => send({ type: 'pause' }), [send]);
  const reset = useCallback(() => send({ type: 'reset' }), [send]);

  // ---- 回调注册 ----
  const onInit = useCallback(
    (fn: (data: InitData) => void) => {
      listenersRef.current.init.push(fn);
      return () => {
        const idx = listenersRef.current.init.indexOf(fn);
        if (idx >= 0) listenersRef.current.init.splice(idx, 1);
      };
    },
    [],
  );

  const onEpoch = useCallback(
    (fn: (ev: EpochData) => void) => {
      listenersRef.current.epoch.push(fn);
      return () => {
        const idx = listenersRef.current.epoch.indexOf(fn);
        if (idx >= 0) listenersRef.current.epoch.splice(idx, 1);
      };
    },
    [],
  );

  const onDone = useCallback(
    (fn: () => void) => {
      listenersRef.current.done.push(fn);
      return () => {
        const idx = listenersRef.current.done.indexOf(fn);
        if (idx >= 0) listenersRef.current.done.splice(idx, 1);
      };
    },
    [],
  );

  const onReset = useCallback(
    (fn: () => void) => {
      listenersRef.current.reset.push(fn);
      return () => {
        const idx = listenersRef.current.reset.indexOf(fn);
        if (idx >= 0) listenersRef.current.reset.splice(idx, 1);
      };
    },
    [],
  );

  return {
    get params() { return stateRef.current.params; },
    get history() { return stateRef.current.history; },
    get dataset() { return stateRef.current.dataset; },
    get lastEvent() { return stateRef.current.lastEvent; },
    step,
    play,
    pause,
    reset,
    onInit,
    onEpoch,
    onDone,
    onReset,
  };
}

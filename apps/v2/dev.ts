/**
 * v2 dev launcher — starts WS server + Next.js, kills both on exit.
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { resolve } from 'node:path';

const procs: ChildProcess[] = [];

function cleanup() {
  for (const p of procs) {
    try { process.kill(-p.pid!, 'SIGTERM'); } catch {}
  }
}

process.on('SIGINT', () => { cleanup(); process.exit(0); });
process.on('SIGTERM', () => { cleanup(); process.exit(0); });
process.on('exit', cleanup);

const opts = { stdio: 'inherit' as const, detached: true };

const server = spawn('npx', ['tsx', resolve('apps/v2/server.ts')], opts);
server.on('exit', (code) => {
  if (code !== 0 && code !== null) { cleanup(); process.exit(code ?? 1); }
});
procs.push(server);

const next = spawn('npx', ['next', 'dev', '--port', '3002'], { ...opts, cwd: resolve('apps/v2') });
next.on('exit', (code) => { cleanup(); process.exit(code ?? 0); });
procs.push(next);

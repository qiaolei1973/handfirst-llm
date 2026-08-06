/**
 * v3 dev launcher — starts WS server + Next.js, kills both on exit.
 *
 * Spawns commands directly (no pnpm wrapper) so kill() reaches the actual process.
 * Usage: pnpm exec tsx apps/v3/dev.ts
 */
import { spawn, type ChildProcess } from 'node:child_process';
import { resolve } from 'node:path';

const procs: ChildProcess[] = [];

function cleanup() {
  for (const p of procs) {
    // kill process group so grandchildren (tsx, next) are also killed
    try { process.kill(-p.pid!, 'SIGTERM'); } catch {}
  }
}

process.on('SIGINT', () => { cleanup(); process.exit(0); });
process.on('SIGTERM', () => { cleanup(); process.exit(0); });
process.on('exit', cleanup);

const opts = { stdio: 'inherit' as const, detached: true };

// WS server
const server = spawn('npx', ['tsx', resolve('apps/v3/server.ts')], opts);
server.on('exit', (code) => {
  if (code !== 0 && code !== null) {
    cleanup(); process.exit(code ?? 1);
  }
});
procs.push(server);

// Next.js
const next = spawn('npx', ['next', 'dev', '--port', '3003'], {
  ...opts,
  cwd: resolve('apps/v3'),
});
next.on('exit', (code) => { cleanup(); process.exit(code ?? 0); });
procs.push(next);

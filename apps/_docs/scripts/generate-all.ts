#!/usr/bin/env tsx
/**
 * generate-all.ts — 生成所有图片（matplotlib / excalidraw）
 *
 * 缓存：对每个源文件存 sha256，hash 不变 → 跳过
 * 强制重生成: tsx generate-all.ts --force  或  GENERATE_ALL_FORCE=1 tsx generate-all.ts
 *
 * 调用方：_docs/package.json 的 predev / prebuild hooks
 */

import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import excalirender from "excalirender";

// ── configuration ──

const PUBLIC = resolve(import.meta.dirname!, "..", "public");
const SCRIPTS = import.meta.dirname!;
const CACHE = join(PUBLIC, ".cache");
const FORCE = process.env["GENERATE_ALL_FORCE"] === "1" || process.argv.includes("--force");

// ── helpers ──

function sha256(file: string): string {
  return createHash("sha256").update(readFileSync(file)).digest("hex");
}

function sh(command: string, cwd?: string) {
  execSync(command, { cwd, stdio: "pipe" });
}

/** Returns true if the source file has changed since last build (or cache missing). */
function cacheMiss(vn: string, src: string): boolean {
  if (FORCE) return true;

  const key = `${vn}/${basename(src)}`;
  const cacheFile = join(CACHE, key);
  const hash = sha256(src);

  if (existsSync(cacheFile) && readFileSync(cacheFile, "utf-8") === hash) {
    return false;
  }

  // Write new hash
  mkdirSync(dirname(cacheFile), { recursive: true });
  writeFileSync(cacheFile, hash);
  return true;
}

/** Remove cache entries whose source files no longer exist. */
function cleanStale() {
  if (!existsSync(CACHE)) return;

  for (const vn of readdirSync(CACHE)) {
    const vnDir = join(CACHE, vn);
    if (!existsSync(vnDir)) continue;
    for (const file of readdirSync(vnDir)) {
      const src = join(SCRIPTS, vn, file);
      if (!existsSync(src)) {
        rmSync(join(vnDir, file));
      }
    }
    // Remove empty vn dirs
    if (readdirSync(vnDir).length === 0) {
      rmSync(vnDir, { recursive: true });
    }
  }
}

// ── sources ──

const VERSIONS = ["v1", "v2", "v3", "v4", "recap"] as const;

type Source = {
  vn: string;
  src: string; // absolute path
  build(): Promise<void>;
};

function collect(): Source[] {
  const sources: Source[] = [];

  for (const vn of VERSIONS) {
    const dir = join(SCRIPTS, vn);
    const out = join(PUBLIC, vn);

    // Python scripts
    for (const f of listPy(dir)) {
      sources.push({
        vn,
        src: join(dir, f),
        async build() {
          sh(`python3 "${join(dir, f)}"`);
        },
      });
    }

    // Excalidraw scripts
    for (const f of listExcalidraw(dir)) {
      const name = basename(f, ".excalidraw");
      sources.push({
        vn,
        src: join(dir, f),
        async build() {
          await excalirender(join(dir, f), { output: join(out, `${name}.svg`) });
        },
      });
    }
  }

  return sources;
}

function listPy(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".py") && f !== "precompute.py" && f !== "generate-steps.py")
    .sort();
}

function listExcalidraw(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".excalidraw") && f !== "backprop-steps.excalidraw")
    .sort();
}

// ── step diagram generator ──
// backprop-steps.excalidraw is the master; generate-step variants from it.

function generateSteps() {
  const master = join(SCRIPTS, "v3", "backprop-steps.excalidraw");
  if (!existsSync(master)) return;
  if (!cacheMiss("v3", master)) {
    console.log("  v3/backprop-steps  (cached)");
    return;
  }
  console.log("  v3/backprop-steps → step variants");
  execSync(`python3 "${join(SCRIPTS, "v3", "generate-steps.py")}"`, { stdio: "pipe" });
}

// ── main ──

console.log("==> Generate images...");

// Step 0: generate progressive step variants from the master
generateSteps();

const sources = collect();

for (const s of sources) {
  const label = `${s.vn}/${basename(s.src)}`;

  if (cacheMiss(s.vn, s.src)) {
    console.log(`  ${label}`);
    try {
      await s.build();
    } catch (err: any) {
      console.error(`  ✗ ${label} failed:`);
      console.error(err.stderr?.toString() ?? String(err));
      process.exit(1);
    }
  } else {
    console.log(`  ${label}  (cached)`);
  }
}

cleanStale();

console.log("==> Done.");

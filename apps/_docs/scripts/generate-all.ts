#!/usr/bin/env tsx
/**
 * generate-all.ts — 生成所有图片（数据图用 matplotlib，架构图用 D2）
 *
 * 缓存：对每个源文件存 sha256，hash 不变 → 跳过
 * 强制重生成: GENERATE_ALL_FORCE=1 tsx generate-all.ts
 *
 * 调用方：_docs/package.json 的 predev / prebuild hooks
 */

import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { homedir } from "node:os";

// ── configuration ──

const PUBLIC = resolve(import.meta.dirname!, "..", "public");
const SCRIPTS = import.meta.dirname!;
const D2 = join(homedir(), ".local", "bin", "d2");
const CACHE = join(PUBLIC, ".cache");
const FORCE = process.env["GENERATE_ALL_FORCE"] === "1";

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
  build(): void;
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
        build() {
          sh(`python3 "${join(dir, f)}"`);
        },
      });
    }

    // D2 scripts
    for (const f of listD2(dir)) {
      const name = basename(f, ".d2");
      sources.push({
        vn,
        src: join(dir, f),
        build() {
          sh(`"${D2}" "${join(dir, f)}" "${join(out, name)}.svg"`);
        },
      });
    }

    // Excalidraw scripts
    for (const f of listExcalidraw(dir)) {
      const name = basename(f, ".excalidraw");
      sources.push({
        vn,
        src: join(dir, f),
        build() {
          sh(`npx tsx "${join(SCRIPTS, "export-excalidraw.ts")}" "${join(dir, f)}" -o "${join(out, name)}.svg"`);
        },
      });
    }
  }

  return sources;
}

function listPy(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".py") && basename(f, ".py") !== "precompute")
    .sort();
}

function listD2(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter((f) => f.endsWith(".d2")).sort();
}

function listExcalidraw(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter((f) => f.endsWith(".excalidraw")).sort();
}

// ── main ──

const sources = collect();

console.log("==> Generate images...");

for (const s of sources) {
  const label = `${s.vn}/${basename(s.src)}`;

  if (cacheMiss(s.vn, s.src)) {
    console.log(`  ${label}`);
    try {
      s.build();
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

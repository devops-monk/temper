// Collects the licence and NOTICE texts of everything compiled into the installers: the Rust crates
// in the desktop shell and the npm packages in the interface (fonts and icons included). Writes
// src/licenses.json, which Settings → About → Open-source licences shows. Run `npm run licenses`
// after changing dependencies; CI fails when the file is out of date.
//
// Tools Temper downloads on the user's computer at runtime (Python packages, llama.cpp) are not
// redistributed by us, so they are listed by name in the dialog instead.

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import spdx from "spdx-license-list/full.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
// The installers' targets (.github/workflows/release.yml); crates for other platforms are never shipped.
const TARGETS = ["aarch64-apple-darwin", "x86_64-apple-darwin", "x86_64-pc-windows-msvc", "aarch64-pc-windows-msvc", "x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"];
const LICENSE_FILE = /^(licen[cs]e|copying|notice|unlicense)([-._].*)?$/i;

/** Licence-like files in a package folder, MIT preferred over Apache for dual-licensed packages. */
function licenseTexts(dir, expression) {
  if (!existsSync(dir)) return [];
  const files = readdirSync(dir).filter((f) => LICENSE_FILE.test(f) && !/\.(rs|js|ts|json|html)$/i.test(f)).sort();
  const notices = files.filter((f) => /^notice/i.test(f));
  let licenses = files.filter((f) => !/^notice/i.test(f));
  // "MIT OR Apache-2.0": either licence is enough, so keep the shorter one.
  if (licenses.length > 1 && /\bOR\b|\//.test(expression ?? "")) {
    const mit = licenses.find((f) => /mit/i.test(f));
    if (mit) licenses = [mit];
  }
  return [...licenses, ...notices].map((f) => readFileSync(join(dir, f), "utf8").replace(/\r\n/g, "\n").trim());
}

function rustPackages() {
  const found = new Map();
  for (const target of TARGETS) {
    const meta = JSON.parse(
      execFileSync("cargo", ["metadata", "--format-version", "1", "--locked", "--filter-platform", target, "--manifest-path", join(root, "src-tauri/Cargo.toml")], {
        maxBuffer: 256 * 1024 * 1024,
      }),
    );
    const byId = new Map(meta.packages.map((p) => [p.id, p]));
    const nodes = new Map(meta.resolve.nodes.map((n) => [n.id, n]));
    // Follow normal dependencies from the app crate; build scripts and dev-dependencies are not shipped.
    const seen = new Set();
    const stack = [meta.resolve.root];
    while (stack.length) {
      const id = stack.pop();
      if (seen.has(id)) continue;
      seen.add(id);
      for (const dep of nodes.get(id)?.deps ?? []) {
        if (dep.dep_kinds.some((k) => k.kind === null)) stack.push(dep.pkg);
      }
    }
    seen.delete(meta.resolve.root);
    for (const id of seen) {
      const p = byId.get(id);
      found.set(id, { name: p.name, version: p.version, license: p.license ?? "", url: p.repository ?? `https://crates.io/crates/${p.name}`, dir: dirname(p.manifest_path) });
    }
  }
  return [...found.values()];
}

/** The standard text of a declared licence, for packages that ship no licence file. */
function standardText(expression) {
  const ids = expression.split(/\s+(?:OR|AND)\s+|\//).map((s) => s.replace(/[()]/g, "").trim());
  const pick = ["MIT", "Apache-2.0", ...ids].find((id) => ids.includes(id) && spdx[id]);
  return pick ? { license: pick, text: spdx[pick].licenseText.trim() } : null;
}

function npmPackages() {
  const out = execFileSync("npm", ["ls", "--omit=dev", "--all", "--parseable", "--userconfig", join(root, ".npmrc")], { cwd: root, maxBuffer: 64 * 1024 * 1024 })
    .toString()
    .trim()
    .split("\n")
    .slice(1); // the first line is Temper itself
  const seen = new Set();
  return out.flatMap((dir) => {
    const pkg = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
    const key = `${pkg.name}@${pkg.version}`;
    if (seen.has(key)) return [];
    seen.add(key);
    const license = typeof pkg.license === "string" ? pkg.license : (pkg.license?.type ?? "");
    const repo = typeof pkg.repository === "string" ? pkg.repository : pkg.repository?.url;
    const url = (repo ?? `https://www.npmjs.com/package/${pkg.name}`).replace(/^git\+/, "").replace(/\.git$/, "").replace(/^github:/, "https://github.com/");
    return [{ name: pkg.name, version: pkg.version, license, url, dir }];
  });
}

const groups = new Map();
const missing = [];
for (const [ecosystem, packages] of [
  ["crate", rustPackages()],
  ["npm", npmPackages()],
]) {
  for (const p of packages) {
    const texts = licenseTexts(p.dir, p.license);
    const entry = { name: p.name, version: p.version, ecosystem, url: p.url };
    if (!texts.length) {
      missing.push({ ...entry, license: p.license });
      continue;
    }
    const text = texts.join("\n\n---\n\n");
    const key = text.replace(/\s+/g, " ");
    if (!groups.has(key)) groups.set(key, { license: p.license, text, packages: [] });
    groups.get(key).packages.push(entry);
  }
}

const sortPackages = (list) => list.sort((a, b) => a.name.localeCompare(b.name) || a.version.localeCompare(b.version));
const result = {
  // Packages that ship no licence file, grouped by their declared licence with its standard text.
  declared: Object.entries(Object.groupBy(missing, (p) => standardText(p.license)?.license ?? (p.license || "Unknown")))
    .map(([license, packages]) => ({ license, text: (spdx[license]?.licenseText.trim() ?? "").replace(/^.*<year>.*$/m, "Copyright (c) the authors of each package listed here."), packages: sortPackages(packages.map(({ license: _, ...p }) => p)) }))
    .sort((a, b) => a.license.localeCompare(b.license)),
  groups: [...groups.values()]
    .map((g) => ({ ...g, packages: sortPackages(g.packages) }))
    .sort((a, b) => a.packages[0].name.localeCompare(b.packages[0].name)),
};

writeFileSync(join(root, "src/licenses.json"), JSON.stringify(result) + "\n");
const count = result.groups.reduce((n, g) => n + g.packages.length, 0) + missing.length;
console.log(`${count} packages, ${result.groups.length} licence texts, ${missing.length} without a licence file → src/licenses.json`);

// Checks the string catalogs in src/locales: every language has exactly English's keys, with the same
// {placeholders}, and every t("key") in the source exists. Run by `npm run check:locales` and CI.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const dir = "src/locales";
const read = (name) => JSON.parse(readFileSync(join(dir, name), "utf8"));
const en = read("en.json");
const holes = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(",");
const problems = [];

for (const file of readdirSync(dir).filter((f) => f.endsWith(".json") && f !== "en.json")) {
  const other = read(file);
  for (const key of Object.keys(en)) {
    if (!(key in other)) problems.push(`${file}: missing "${key}"`);
    else if (holes(other[key]) !== holes(en[key])) problems.push(`${file}: "${key}" has {${holes(other[key])}}, English has {${holes(en[key])}}`);
    else if (!other[key].trim()) problems.push(`${file}: "${key}" is empty`);
  }
  for (const key of Object.keys(other)) if (!(key in en)) problems.push(`${file}: "${key}" isn't in en.json`);
}

function* sources(folder) {
  for (const name of readdirSync(folder)) {
    const path = join(folder, name);
    if (statSync(path).isDirectory()) yield* sources(path);
    else if (/\.tsx?$/.test(name)) yield path;
  }
}
for (const path of sources("src")) {
  for (const m of readFileSync(path, "utf8").matchAll(/\bt\(\s*"([\w.]+)"/g)) if (!(m[1] in en)) problems.push(`${path}: t("${m[1]}") isn't in en.json`);
}

if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log(`Locales match: ${Object.keys(en).length} strings in ${readdirSync(dir).length} languages.`);

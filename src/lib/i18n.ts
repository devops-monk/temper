import { useCallback } from "react";
import { create } from "zustand";
import en from "../locales/en.json";
import es from "../locales/es.json";
import hi from "../locales/hi.json";
import ja from "../locales/ja.json";
import zh from "../locales/zh.json";

/** A string's id in src/locales/en.json; scripts/check-locales.mjs keeps the other languages in step. */
export type Key = keyof typeof en;
export type Language = "en" | "hi" | "es" | "zh" | "ja";

export const LANGUAGES: { id: Language; name: string }[] = [
  { id: "en", name: "English" },
  { id: "hi", name: "हिन्दी" },
  { id: "es", name: "Español" },
  { id: "zh", name: "中文（简体）" },
  { id: "ja", name: "日本語" },
];

const CATALOGS: Record<Language, Partial<Record<Key, string>>> = { en, hi, es, zh, ja };
const KEY = "temper.language";

function initial(): Language {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved && saved in CATALOGS) return saved as Language;
  } catch {
    // no storage: follow the system
  }
  const system = (typeof navigator !== "undefined" ? navigator.language : "en").slice(0, 2);
  return system in CATALOGS ? (system as Language) : "en";
}

export const useLanguage = create<{ language: Language; setLanguage: (l: Language) => void }>((set) => ({
  language: initial(),
  setLanguage: (language) => {
    try {
      localStorage.setItem(KEY, language);
    } catch {
      // kept for this session only
    }
    document.documentElement.lang = language;
    set({ language });
  },
}));
if (typeof document !== "undefined") document.documentElement.lang = useLanguage.getState().language;

export type Vars = Record<string, string | number>;

/** The string in `language`, falling back to English, with {name} placeholders filled from `vars`. */
export function translate(language: Language, key: Key, vars?: Vars): string {
  const text = CATALOGS[language][key] ?? en[key] ?? key;
  return vars ? text.replace(/\{(\w+)\}/g, (hole, name) => (name in vars ? String(vars[name]) : hole)) : text;
}

/** For code outside React; components use useT() so they update when the language changes. */
export const t = (key: Key, vars?: Vars) => translate(useLanguage.getState().language, key, vars);

export function useT() {
  const language = useLanguage((s) => s.language);
  return useCallback((key: Key, vars?: Vars) => translate(language, key, vars), [language]);
}

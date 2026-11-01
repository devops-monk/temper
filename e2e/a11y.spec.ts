import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const PAGES = ["chat", "research", "models", "library", "train", "datasets", "recipes", "export", "connect", "settings"];

for (const theme of ["light", "dark"] as const) {
  test(`every page passes the accessibility checks in the ${theme} theme`, async ({ page }) => {
    await page.addInitScript((t) => {
      localStorage.setItem("temper.tour", "1");
      localStorage.setItem("temper.theme", t);
    }, theme);
    const found: string[] = [];
    for (const name of PAGES) {
      await page.goto(`/#${name}`);
      await expect(page.getByRole("navigation", { name: "Main" })).toBeVisible();
      await page.waitForTimeout(600); // let the page's data arrive
      const { violations } = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
      for (const v of violations) {
        found.push(`${name}: ${v.id} (${v.impact}) ${v.help}\n    ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join("\n    ")}`);
      }
    }
    expect(found, found.join("\n")).toEqual([]);
  });
}

test("the app can be used with the keyboard alone", async ({ page }) => {
  await page.goto("/");
  // The tour takes focus, keeps Tab inside itself, and closes with Escape.
  const tour = page.getByRole("dialog");
  await expect(tour).toBeFocused();
  for (let i = 0; i < 4; i++) await page.keyboard.press("Tab");
  expect(await tour.evaluate((d) => d.contains(document.activeElement))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(tour).toBeHidden();

  // Tab reaches the sidebar, and Enter opens a page.
  const models = page.getByRole("navigation", { name: "Main" }).getByRole("button", { name: "Models", exact: true });
  for (let i = 0; i < 15 && !(await models.evaluate((b) => b === document.activeElement)); i++) await page.keyboard.press("Tab");
  await expect(models).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(models).toHaveAttribute("aria-current", "page");
  // Focus is visible: the outline the app draws for keyboard focus.
  expect(await models.evaluate((b) => getComputedStyle(b).outlineStyle)).toBe("solid");
});

test("pages full of data pass the accessibility checks in both themes", async ({ page, request }) => {
  test.setTimeout(90_000); // eight full-page checks
  // Data like a used Temper's: chats in a folder and pinned, with thinking and a tool call, an API key, and a cloud provider.
  const engine = "http://127.0.0.1:18765";
  const auth = { Authorization: "Bearer e2e" };
  const { createServer } = await import("node:http");
  const provider = createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ data: ["model-a", "model-b", "model-c"].map((id) => ({ id })) }));
  });
  await new Promise<void>((ok) => provider.listen(18797, "127.0.0.1", ok));
  await request.post(`${engine}/api/chats/folders/add`, { headers: auth, data: { name: "Work" } });
  const made: string[] = [];
  for (let i = 0; i < 6; i++) {
    const chat = await (
      await request.post(`${engine}/api/chats`, {
        headers: auth,
        data: {
          title: `Seeded chat ${i}`,
          folder: i < 2 ? "Work" : null,
          messages: [
            { role: "user", content: `Question ${i}: how do I sort a list in Python?` },
            { role: "assistant", content: "Use `sorted(items)` for a new list, or `items.sort()` in place.\n\n```python\nprint(sorted([3, 1, 2]))\n```", reasoning: "They want the simplest answer.", model: "seeded-model", tools: [{ id: "t1", name: "run_python", arguments: { code: "print(sorted([3, 1, 2]))" }, result: "[1, 2, 3]", at: 0 }] },
          ],
        },
      })
    ).json();
    made.push(chat.id);
  }
  await request.patch(`${engine}/api/chats/${made[5]}`, { headers: auth, data: { pinned: true } });
  const key = await (await request.post(`${engine}/api/keys`, { headers: auth, data: { name: "seeded laptop" } })).json();
  const added = await (await request.post(`${engine}/api/providers`, { headers: auth, data: { kind: "custom", key: "sk-seeded-0123456789", name: "seeded", url: "http://127.0.0.1:18797/v1" } })).json();

  try {
    for (const theme of ["light", "dark"] as const) {
      await page.addInitScript((t) => {
        localStorage.setItem("temper.tour", "1");
        localStorage.setItem("temper.theme", t);
      }, theme);
      const found: string[] = [];
      for (const name of ["chat", "connect", "models", "settings"]) {
        await page.goto(`/#${name}`);
        await expect(page.getByRole("navigation", { name: "Main" })).toBeVisible();
        if (name === "chat") await page.getByText("Seeded chat 3").click(); // a thread with thinking, code and a tool call
        await page.waitForTimeout(700);
        const { violations } = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
        for (const v of violations) found.push(`${theme} ${name}: ${v.id} (${v.impact}) ${v.help}\n    ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join("\n    ")}`);
      }
      expect(found, found.join("\n")).toEqual([]);
    }
  } finally {
    for (const id of made) await request.delete(`${engine}/api/chats/${id}`, { headers: auth });
    await request.post(`${engine}/api/chats/folders/remove`, { headers: auth, data: { name: "Work" } });
    await request.delete(`${engine}/api/keys/${key.id}`, { headers: auth });
    await request.delete(`${engine}/api/providers/${added.id}`, { headers: auth });
    provider.close();
  }
});

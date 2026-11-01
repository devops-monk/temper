import { expect, type Page, test } from "@playwright/test";

/** Skips the first-launch tour, for the tests that aren't about it. */
async function seenTour(page: Page) {
  await page.addInitScript(() => localStorage.setItem("temper.tour", "1"));
}

/** Fails a test on an error the interface didn't catch, or a page that failed to draw. */
function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

test("a first launch shows the tour, which can be walked and skipped for good", async ({ page }) => {
  await page.goto("/");
  const tour = page.getByRole("dialog");
  await expect(tour.getByRole("heading", { name: "Welcome to Temper" })).toBeVisible();
  await tour.getByRole("button", { name: "Show me" }).click();
  await expect(tour.getByRole("heading", { name: "Models" })).toBeVisible();
  await page.keyboard.press("ArrowRight");
  await expect(tour.getByRole("heading", { name: "Chat" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(tour).toBeHidden();
  await page.reload();
  await expect(page.getByRole("navigation", { name: "Main" })).toBeVisible();
  await expect(page.getByRole("dialog")).toBeHidden();
});

test("every page opens without an error", async ({ page }) => {
  await seenTour(page);
  const errors = watchErrors(page);
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Main" });
  for (const name of ["Chat", "Research", "Models", "Library", "Train", "Datasets", "Recipes", "Export", "Connect", "Settings"]) {
    await nav.getByRole("button", { name, exact: true }).click();
    await expect(nav.getByRole("button", { name, exact: true })).toHaveAttribute("aria-current", "page");
    await expect(page.getByText("This page hit an error")).toBeHidden();
  }
  expect(errors).toEqual([]);
});

test("Connect makes an API key that opens the model API, then revokes it", async ({ page, request }) => {
  await seenTour(page);
  await page.goto("/#connect");
  await page.getByLabel("New key's name").fill("e2e laptop");
  await page.getByRole("button", { name: "Create key" }).click();
  const shown = page.getByText(/^tk-[A-Za-z0-9_-]{40,}$/); // the whole key, not the table's short hint
  await expect(shown).toBeVisible();
  const key = (await shown.textContent())!.trim();

  const models = await request.get("http://127.0.0.1:18765/v1/models", { headers: { Authorization: `Bearer ${key}` } });
  expect(models.status()).toBe(200);
  // The request shows up in the monitor, under the key's name.
  await expect(page.getByRole("cell", { name: "GET /v1/models" }).first()).toBeVisible({ timeout: 10_000 });

  page.once("dialog", (d) => void d.accept());
  await page.getByRole("button", { name: "Revoke e2e laptop" }).click();
  await expect(page.getByRole("button", { name: "Revoke e2e laptop" })).toBeHidden();
  const after = await request.get("http://127.0.0.1:18765/v1/models", { headers: { Authorization: `Bearer ${key}` } });
  expect(after.status()).toBe(401);
});

test("Settings shows statistics and privacy, with crash reports off until turned on", async ({ page }) => {
  await seenTour(page);
  await page.goto("/#settings");
  await expect(page.getByRole("heading", { name: "Statistics" })).toBeVisible();
  await expect(page.getByText("tokens written", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Privacy" })).toBeVisible();
  await expect(page.getByText("Temper sends no telemetry")).toBeVisible();
  const offer = page.getByRole("switch", { name: "Offer to report crashes" });
  await expect(offer).toHaveAttribute("aria-checked", "false");
  await page.getByRole("button", { name: "Take the tour" }).click();
  await expect(page.getByRole("dialog").getByRole("heading", { name: "Welcome to Temper" })).toBeVisible();
});

test("a cloud provider's model answers in chat", async ({ page }) => {
  // A stand-in for an OpenAI-compatible provider, answering with the key it was given.
  const { createServer } = await import("node:http");
  const seen: { auth?: string; model?: string }[] = [];
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      if (req.headers.authorization !== "Bearer sk-e2e-0123456789") return void res.writeHead(401).end();
      if (req.url === "/v1/models") return void res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ data: [{ id: "cloud-model" }] }));
      const ask = JSON.parse(body);
      seen.push({ auth: req.headers.authorization, model: ask.model });
      const chunk = (d: object) => `data: ${JSON.stringify({ choices: [{ index: 0, delta: d }] })}\n\n`;
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.end(chunk({ role: "assistant", content: "Hello from " }) + chunk({ content: "the cloud." }) + "data: [DONE]\n\n");
    });
  });
  await new Promise<void>((ok) => server.listen(18799, "127.0.0.1", ok));
  try {
    await seenTour(page);
    const errors = watchErrors(page);
    await page.goto("/#connect");
    await page.getByLabel("Provider", { exact: true }).selectOption("custom");
    await page.getByLabel("API key", { exact: true }).fill("sk-e2e-0123456789");
    await page.getByLabel("Name", { exact: true }).fill("mock");
    await page.getByLabel("The API's address").fill("http://127.0.0.1:18799/v1");
    await page.getByRole("button", { name: "Add", exact: true }).click();
    await expect(page.getByText("Connected")).toBeVisible();
    await page.getByRole("button", { name: "Use in chat" }).first().click();
    const box = page.getByPlaceholder("Message the model");
    await box.fill("hi");
    await box.press("Enter");
    await expect(page.getByText("Hello from the cloud.")).toBeVisible({ timeout: 15_000 });
    expect(seen.at(-1)?.model).toBe("cloud-model");
    expect(errors).toEqual([]);
  } finally {
    server.close();
  }
});

test("Settings adds an admin and a member, who signs in to their own chats", async ({ page, request }) => {
  await seenTour(page);
  const errors = watchErrors(page);
  await page.goto("/#settings");
  await page.getByLabel("Name", { exact: true }).fill("ann");
  await page.getByLabel("Password, 8 or more characters").fill("password1");
  await page.getByRole("button", { name: "Add the admin" }).click();
  await expect(page.getByText("ann", { exact: true })).toBeVisible();
  await page.getByLabel("Name", { exact: true }).fill("bob");
  await page.getByLabel("Password, 8 or more characters").fill("password2");
  await page.getByRole("button", { name: "Add account" }).click();
  await expect(page.getByText("bob", { exact: true })).toBeVisible();

  const engine = "http://127.0.0.1:18765";
  const login = await request.post(`${engine}/api/accounts/login`, { data: { name: "bob", password: "password2" } });
  const bob = { Authorization: `Bearer ${(await login.json()).token}` };
  await request.post(`${engine}/api/chats`, { headers: bob, data: { title: "bob's chat" } });
  const theirs = await (await request.get(`${engine}/api/chats`, { headers: bob })).json();
  expect(theirs.chats.map((c: { title: string }) => c.title)).toEqual(["bob's chat"]);
  expect((await request.get(`${engine}/api/keys`, { headers: bob })).status()).toBe(403);
  const owners = await (await request.get(`${engine}/api/chats`, { headers: { Authorization: "Bearer e2e" } })).json();
  expect(owners.chats.map((c: { title: string }) => c.title)).not.toContain("bob's chat");

  // Back to no accounts, for the next run's first launch.
  page.on("dialog", (d) => void d.accept());
  await page.getByRole("button", { name: "Remove bob" }).click();
  await expect(page.getByRole("button", { name: "Remove bob" })).toBeHidden();
  expect(errors).toEqual([]);
});

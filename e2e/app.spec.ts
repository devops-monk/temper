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

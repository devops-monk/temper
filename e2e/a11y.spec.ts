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

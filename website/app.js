// Fade sections in as they enter the viewport, scale the hero screenshot up with scroll,
// and switch the nav to its light style over light bands. All of it is skipped for reduced motion.
(() => {
  document.documentElement.classList.remove("no-js");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fades = document.querySelectorAll("[data-fade]");

  if (reduce || !("IntersectionObserver" in window)) {
    fades.forEach((el) => el.classList.add("in"));
  } else {
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            io.unobserve(entry.target);
          }
        }
      },
      { rootMargin: "0px 0px -10% 0px" },
    );
    fades.forEach((el) => io.observe(el));
  }

  for (const button of document.querySelectorAll("[data-copy]")) {
    button.addEventListener("click", async () => {
      const text = button.parentElement.querySelector("code").textContent;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch {
        button.textContent = "Selected";
        getSelection().selectAllChildren(button.parentElement.querySelector("code"));
      }
      setTimeout(() => (button.textContent = "Copy"), 1600);
    });
  }

  const nav = document.getElementById("nav");
  const bands = [...document.querySelectorAll("main > section")];
  const shot = document.querySelector("[data-rise]");

  let ticking = false;
  function update() {
    ticking = false;
    const probe = nav.offsetHeight / 2;
    const band = bands.find((b) => {
      const r = b.getBoundingClientRect();
      return r.top <= probe && r.bottom > probe;
    });
    nav.classList.toggle("on-light", !!band && !band.classList.contains("dark"));

    if (shot && !reduce) {
      // From 92% to full size over the first screen of scrolling.
      const progress = Math.min(1, Math.max(0, window.scrollY / (window.innerHeight * 0.9)));
      shot.style.transform = `perspective(1600px) rotateX(${(1 - progress) * 8}deg) scale(${0.92 + progress * 0.08})`;
    }
  }
  addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(update);
      }
    },
    { passive: true },
  );
  addEventListener("resize", update);
  update();
})();

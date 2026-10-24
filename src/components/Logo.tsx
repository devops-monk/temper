/** The "T" blade from the app icon, without the tile. */
export function LogoMark({ className = "size-6" }: { className?: string }) {
  return (
    <svg viewBox="250 240 524 590" className={className} aria-hidden>
      <defs>
        <linearGradient id="logo-temper" x1="0" y1="250" x2="0" y2="820" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#F3DD9A" />
          <stop offset=".28" stopColor="#D9A55C" />
          <stop offset=".52" stopColor="#B0683E" />
          <stop offset=".72" stopColor="#8459B8" />
          <stop offset="1" stopColor="#3A7BDB" />
        </linearGradient>
      </defs>
      <path
        d="M300 250 H724 a42 42 0 0 1 0 84 H568 V700 L512 820 L456 700 V334 H300 a42 42 0 0 1 0 -84 Z"
        fill="url(#logo-temper)"
      />
    </svg>
  );
}

import React from "react";

/**
 * The signature element: a camera-aperture mark built from overlapping
 * blades, literalizing "Lens" in CareerLens — narrowing down a wide field
 * of careers to a single point of focus. Used as the static logo mark and,
 * with `spinning`, as the analysis-in-progress state.
 */
export default function ApertureMark({ size = 40, spinning = false, blades = 7 }) {
  const bladeArr = Array.from({ length: blades });
  const radius = size / 2;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      className={spinning ? "aperture spinning" : "aperture"}
      role="img"
      aria-label="CareerLens"
    >
      <circle cx="50" cy="50" r="48" fill="none" stroke="var(--hairline)" strokeWidth="1" />
      <g style={{ transformOrigin: "50px 50px" }}>
        {bladeArr.map((_, i) => {
          const angle = (360 / blades) * i;
          return (
            <path
              key={i}
              d="M50 50 L50 6 A44 44 0 0 1 78 20 Z"
              fill="var(--amber)"
              opacity={spinning ? 0.55 : 0.85}
              transform={`rotate(${angle} 50 50)`}
            />
          );
        })}
      </g>
      <circle cx="50" cy="50" r="14" fill="var(--ink)" stroke="var(--amber)" strokeWidth="2" />
    </svg>
  );
}

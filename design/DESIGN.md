# Design System Strategy: Enterprise Precision & Tonal Depth

## 1. Overview & Creative North Star
This design system is built to transform complex MIC (Microphone) testing data into a high-end, editorial-grade dashboard. The "Creative North Star" is **"The Precision Lab"**—a visual metaphor that combines the cold accuracy of laboratory instrumentation with the sophisticated clarity of modern digital editorial.

Unlike generic enterprise tools that rely on cluttered grids and heavy borders, this system utilizes **Tonal Layering** and **Intentional Asymmetry**. We break the "template" look by using exaggerated typographic scales for key metrics and overlapping surface containers that suggest physical depth. The result is a dashboard that feels less like a spreadsheet and more like a custom-engineered command center.

## 2. Colors
Our palette is rooted in a foundation of deep, trustworthy blues and clinical grays, accented by high-saturation status indicators.

### The "No-Line" Rule
**Explicit Instruction:** Prohibit the use of 1px solid borders for sectioning or card containment. Structural boundaries must be defined exclusively through background color shifts. For example, a `surface-container-low` component should sit on a `surface` background to define its edge. The transition of color is the boundary.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers. Use the surface-container tiers to create an architectural hierarchy:
*   **Base:** `surface` (#f9f9fd)
*   **Primary Work Area:** `surface-container-low` (#f2f3f9)
*   **Data Cards:** `surface-container-lowest` (#ffffff) to provide "pop" against the gray-blue base.
*   **Interactions/Modals:** `surface-bright` (#f9f9fd)

### The "Glass & Gradient" Rule
To elevate the "Precision Lab" aesthetic, floating elements (like tooltips or temporary overlays) should utilize **Glassmorphism**. Use `surface` at 80% opacity with a `backdrop-blur` of 12px.
*   **Signature Textures:** For main Action Buttons or Hero Data points (like "Overall Yield"), apply a subtle linear gradient: `primary` (#2559bd) to `primary_dim` (#0f4db0) at a 135-degree angle. This adds "soul" and professional polish.

## 3. Typography
We utilize a dual-typeface system to balance high-impact branding with extreme data density.

*   **Display & Headlines (Manrope):** Used for titles and key instrumentation metrics. Manrope’s geometric nature feels engineered and authoritative.
*   **Body & Labels (Inter):** Used for all data tables, graph axis labels, and technical readouts. Inter is chosen for its superior legibility at small sizes (`label-sm` at 0.6875rem).

**Editorial Hierarchy:**
*   **Metric Value:** `display-sm` (Manrope) - Bold, high-contrast.
*   **Metric Label:** `label-md` (Inter) - Uppercase with 0.05rem letter spacing for a "technical tag" look.
*   **Data Tables:** `body-md` (Inter) for primary data; `body-sm` for secondary metadata.

## 4. Elevation & Depth
Depth is a functional tool for focus, not just a stylistic choice.

*   **The Layering Principle:** Avoid shadows for static cards. Instead, stack `surface-container-lowest` on top of `surface-container-high`.
*   **Ambient Shadows:** Use only for active, draggable, or floating elements.
    *   *Shadow Definition:* `0px 8px 24px rgba(44, 51, 58, 0.06)`. Note the color is a tint of `on-surface`, making it feel like ambient light rather than "ink."
*   **The "Ghost Border" Fallback:** If a chart requires a container for accessibility, use the `outline-variant` token (#acb3bc) at **15% opacity**.
*   **Glassmorphism:** For sidebars and "Control Trays," use a semi-transparent `surface_container_low` with a backdrop blur to maintain context of the underlying data visualizations.

## 5. Components

### Cards & Data Organizers
*   **Rule:** Forbid divider lines.
*   **Implementation:** Separate groups of data using `spacing-8` (1.75rem) or by shifting the background from `surface-container-low` to `surface-container-highest`. Use asymmetric padding (e.g., more padding on the left than the right) to guide the eye toward data trends.

### Buttons & Chips
*   **Primary Button:** Gradient fill (`primary` to `primary_dim`), `lg` roundedness (0.5rem), and white `on-primary` text.
*   **Chips (Status):** For "Pass/Fail" indicators, use high-saturation containers (`error_container` or `primary_container`) with 10% opacity, allowing the background to show through, paired with full-opacity bold text.

### Technical Input Fields
*   **Look:** "Instrumentation Style." Use `surface_container_highest` for the background. No borders. On focus, a bottom-only 2px line in `primary` appears.

### Instrumentation Graphs (Modernized)
*   **Containers:** Modernize existing graphs by removing their individual borders. Place graphs on a clean `surface-container-lowest` card.
*   **Overlays:** Use the `tertiary` color (#605c78) for reference lines (LCL/UCL) to distinguish them from the primary data paths.

## 6. Do's and Don'ts

### Do:
*   **Do** use vertical white space (`spacing-12` or `16`) to separate distinct data modules.
*   **Do** use `manrope` for the numeric values in the "Production Dashboard" to give them a premium, dashboard-hero feel.
*   **Do** use "Ghost Borders" for the table headers to separate the header from the data without cluttering the view.

### Don't:
*   **Don't** use 100% black shadows. They feel "dirty" and "cheap."
*   **Don't** use standard "Success Green" or "Danger Red" if they clash with the blue-gray base. Use the provided `error` (#9f403d) and high-contrast blue `primary` for "Pass" states to maintain the professional palette.
*   **Don't** use more than two levels of nesting. If a card needs a sub-section, use a background shift, not a new card-on-card shadow.
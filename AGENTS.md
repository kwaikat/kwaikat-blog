# Repository Guidelines

## Project Structure & Module Organization
All feature work lives in `src/`: routes inside `src/pages`, shared UI in `src/components` or `src/layouts`, and Markdown/MDX content in `src/content/posts`. Global metadata and theme knobs belong to `src/config.ts`, while shared styling utilities sit in `src/styles`. Static files must be dropped into `public/` so Astro copies them unchanged, and long-form docs stay in `docs/`. Keep automation or build tweaks within `scripts/` (e.g., `scripts/new-post.js`) and the root config files (`astro.config.mjs`, `svelte.config.js`, Tailwind/PostCSS) to make reviews predictable.

## Build, Test, and Development Commands
Use pnpm exclusively. Typical commands:
- `pnpm dev` (`pnpm start`): Astro dev server with hot reload on `localhost:4321`.
- `pnpm build`: compile to `dist/` and run Pagefind indexing for search.
- `pnpm preview`: serve the production build for manual QA.
- `pnpm check`: Astro’s template/runtime validation.
- `pnpm type-check`: TypeScript verification.
- `pnpm format` / `pnpm lint`: Biome formatting and linting for `src/`.
- `pnpm new-post my-title`: create `src/content/posts/my-title.mdx` with the correct frontmatter shell.

## Coding Style & Naming Conventions
Adopt 2-space indentation and ESM syntax everywhere. Components are PascalCase (`HeroBanner.astro`), helpers camelCase (`getPostMeta.ts`), and folders/routes kebab-case (`about`). Frontmatter keys remain lowercase kebab-case (`published`, `image`, `lang`). Prefer named exports, colocate helpers near consumers, and isolate formatting-only commits after running `pnpm format` so reviews stay focused on behavior.

## Testing Guidelines
There is no browser test suite, so treat static analysis as blocking. Before pushing a branch run `pnpm lint`, `pnpm format --check`, `pnpm type-check`, `pnpm check`, and `pnpm build`, then open `pnpm preview` to verify navigation, code blocks, RSS, and Pagefind results. Always create or rename posts via `pnpm new-post` to keep filenames and frontmatter consistent.

## Commit & Pull Request Guidelines
History shows short imperative subjects (`add github pages deploy workflow`, `修改base url`); match that voice and keep each commit single-purpose. Pull requests must describe the change, link issues, list the verification commands, and include screenshots or terminal logs any time UI or build output shifts. Call out edits to `astro.config.mjs`, `src/config.ts`, or deployment workflows so reviewers can assess release impact quickly.

## Security & Configuration Tips
Store secrets and analytics IDs only in ignored `.env` files. Introduce third-party scripts through Astro integrations or head helpers rather than raw `<script>` tags. When the canonical URL or base path changes, update `astro.config.mjs`, `src/config.ts`, `vercel.json`, and GitHub Pages workflows in the same pull request to avoid mismatched routing.

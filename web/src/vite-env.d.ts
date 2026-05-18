/// <reference types="vite/client" />

// Injected by `vite.config.ts` `define`. Resolved at build time from
// `git describe --tags --dirty --always`, falling back to
// `package.json` and finally the literal `dev`.
declare const __APP_VERSION__: string
